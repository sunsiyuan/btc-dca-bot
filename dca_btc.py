#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BTC 定投策略框架（当前版本）
=============================

本系统是一个“估值因子 × 风险因子”的动态定投模型，用于 BTC 的长期累积型投资。
核心目标是用数据量化“买多少”，用机制避免极端风险，并在结构良好的低估周期中显著提高买入力度。

整体架构由三部分组成：

----------------------------------------------------------------------
1. 估值因子（Valuation Factors）→ 决定“价格贵还是便宜”
----------------------------------------------------------------------
估值因子不直接预测未来价格，而是判断当前是否属于历史上偏低/偏高的区域。
使用的输入包括但不限于：
- Mayer Multiple（价格 / 200W MA）
- 距离 200W MA 百分比（价格定位）
- SSR-like（BTC 市值 ÷ 稳定币市值篮子）
- 7 日趋势（短期价格方向）
- 其他可扩展估值指标

这些分项会被加总为一个“估值得分（total score）”，
再映射成一个基础 multiplier（估值倍数）。
估值越低，倍数越大；越高估，倍数越小或为 0。

估值倍数体现的是：
“在这个价格区间，长线配置应当有多积极。”

---------------------------------------------------------------------- 
2. 风险因子（Risk Budget Factor）→ 决定“今天能不能这么 aggressive 买”
----------------------------------------------------------------------
风险因子通过市场结构状态来调节买入强度，避免在高杠杆、高波动的危险时段过度建仓。
使用的输入包括但不限于：
- 30 日年化波动率（Volatility）
- Funding Rate（资金费率）
- 永续合约 OI（杠杆规模）
- OI 的 7 日趋势（杠杆是否在累积）
- 其他可扩展市场结构指标

风险因子会输出一个 0.0–1.0 的连续权重：
- 趋于 1.0 表示“风险并不高，可以安心买”
- 趋于 0.4–0.6 表示“结构不稳，需减少买入”
- 极端风险区间会进一步降低倍数

风险因子的作用是：
“在风险高的区间，即使便宜也要谨慎；风险低时才允许强力加仓。”

---------------------------------------------------------------------- 
3. 最终投入计算 = Base × Multiplier × RiskFactor
----------------------------------------------------------------------
最终每日实际买入金额由三者共同决定：
- Base：基础定投金额（固定设定）
- Multiplier：估值得分映射出的倍数（1.5x / 2.5x / 6x / 10x ...）
- RiskFactor：市场结构健康度（0.4–1.0）

合成行为特征：
- 高估 + 高风险：0× 或极低买入
- 高估 + 低风险：仍然少量买入，避免踏空
- 低估 + 高风险：保持谨慎，只在结构稳定后再加大力度
- 低估 + 低风险：强力加仓，提升每轮熊市底部获取筹码的效率

这是一个“价值驱动 + 风险控制”的组合机制，
避免了纯估值 DCA 的盲目，也避免了情绪化追高。

---------------------------------------------------------------------- 
4. 数据源与 fallback 机制
----------------------------------------------------------------------
为了在 GitHub Actions 中长期稳定运行，所有关键数据都有多层 fallback：
- 价格与 K 线：Binance → OKX
- Funding / OI / Mark Price：Hyperliquid → Binance
- OI 历史：Binance → Hyperliquid 本地缓存（按日去重）
- SSR-like：CoinGecko 稳定币篮子

若实际执行中某数据源失败，则自动回退，保证每日运行不中断。

---------------------------------------------------------------------- 
5. 核心设计理念
----------------------------------------------------------------------
本策略从不预测价格，而是：
- 在“不贵但又安全”的区间多买
- 在“波动+杠杆都在累积”的时期少买
- 在“极端高估或高风险”的区间停止买
- 在“极端低估且风险释放”的区间大力建仓

整个框架旨在提升：
✔ 有效买入质量  
✔ 资金使用效率  
✔ 长期收益/回撤比  
✔ 极端行情下的防御能力  

同时保持代码结构可扩展，可随时添加新的估值因子或风险因子。

----------------------------------------------------------------------
6. 结果输出
----------------------------------------------------------------------
- 指标快照
- 定投倍数
- 总结性建议
----------------------------------------------------------------------
"""

import math
import os
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUT = "https://fapi.binance.com"
OKX_SPOT = "https://www.okx.com"
CG = "https://api.coingecko.com/api/v3"
HL_INFO = "https://api.hyperliquid.xyz/info"
# 将缓存文件放在脚本同目录，避免因工作目录不同导致路径漂移
OI_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hl_oi_history.json")

# 数据源配置（支持多数据源 + 回退）
DATA_SOURCE = os.getenv("DATA_SOURCE", "binance").lower()  # "binance" 或 "okx"
DERIV_SOURCE = os.getenv("DERIV_SOURCE", "hyperliquid").lower()  # "binance" 或 "hyperliquid"
SIMULATE_BINANCE_FAIL = os.getenv("SIMULATE_BINANCE_FAIL", "0") == "1"

# Hyperliquid perp 资产上下文缓存（避免重复请求）
_HL_PERP_CTX = None

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-dca-model/1.0"})

# 记录本次运行实际使用的数据源
DATA_SOURCES = {
    "kline": "",
    "mark_price": "",
    "funding": "",
    "oi": "",
    "oi_history": "",
    "ssr": "",
}

# ---------------------------
# 工具
# ---------------------------
def http_get(url, params=None):
    try:
        r = SESSION.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️ API 调用失败: {url} {e}")
        return None


def http_post_json(url, payload=None):
    try:
        r = SESSION.post(url, json=payload or {}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️ API 调用失败: {url} {e}")
        return None


# ---------------------------
# BTC K线（Binance 主，OKX 现货 BTC-USDT 副）
# ---------------------------
def _get_klines_from_binance(interval: str = "1d", limit: int = 500) -> pd.DataFrame:
    """原始 Binance K 线封装，支持模拟失败，用于本地测试 fallback。"""
    if SIMULATE_BINANCE_FAIL:
        raise RuntimeError("SIMULATE_BINANCE_FAIL=1，模拟 Binance 失败")

    data = http_get(f"{BINANCE_SPOT}/api/v3/klines", {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": limit,
    })
    if not data:
        raise RuntimeError("Binance 返回空数据")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df.set_index("open_time", inplace=True)
    return df


def _get_klines_from_okx(interval: str = "1d", limit: int = 500) -> pd.DataFrame:
    """OKX 版 BTC-USDT K 线，作为 Binance 的二级数据源。"""
    if interval == "1d":
        bar = "1D"
    elif interval == "1w":
        bar = "1W"
    else:
        raise ValueError(f"暂不支持的 interval: {interval}")

    # OKX 接口一次最多返回的条数有限，这里至少拿到 200 根，最多 300 根
    actual_limit = min(max(limit, 200), 300)

    data = http_get(f"{OKX_SPOT}/api/v5/market/candles", {
        "instId": "BTC-USDT",
        "bar": bar,
        "limit": actual_limit,
    })
    if not data or "data" not in data or not data["data"]:
        raise RuntimeError("OKX 返回空数据")

    # OKX 返回的数据是按时间倒序（最近在前），这里倒过来，保证时间从旧到新
    rows = list(reversed(data["data"]))
    df = pd.DataFrame(rows)

    # 约定列：0=ts(ms),1=open,2=high,3=low,4=close,5=volume
    df = df.rename(columns={0: "open_time", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"})
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df.set_index("open_time", inplace=True)
    return df


def get_klines(interval: str = "1d", limit: int = 500) -> pd.DataFrame:
    """对外统一入口：优先尝试 DATA_SOURCE，再自动回退（binance → okx）。"""
    sources = []
    primary = (DATA_SOURCE or "binance").lower()
    if primary == "binance":
        sources = ["binance", "okx"]
    elif primary == "okx":
        sources = ["okx", "binance"]
    else:
        sources = [primary, "binance", "okx"]

    last_err = None
    for src_name in sources:
        try:
            if src_name == "binance":
                DATA_SOURCES["kline"] = "binance"
                return _get_klines_from_binance(interval, limit)
            elif src_name == "okx":
                DATA_SOURCES["kline"] = "okx"
                return _get_klines_from_okx(interval, limit)
        except Exception as e:
            last_err = e
            print(f"⚠️ get_klines 使用 {src_name} 失败: {e}")

    if last_err:
        raise last_err
    return pd.DataFrame()


# ---------------------------
# 指标 — 估值
# ---------------------------
def get_mayer() -> tuple[float, float, float]:
    df = get_klines("1d", 400)
    if len(df) < 200:
        raise ValueError("历史不足 200 天")

    price = df["close"].iloc[-1]
    ma200 = df["close"].tail(200).mean()
    mayer = price / ma200
    return float(price), float(ma200), float(mayer)


def get_200w_distance(price=None) -> tuple[float, float]:
    df = get_klines("1w", 300)
    if len(df) < 200:
        return float("nan"), float("nan")

    ma200w = df["close"].tail(200).mean()
    if price is None:
        price = df["close"].iloc[-1]
    dist = (price - ma200w) / ma200w
    return float(ma200w), float(dist)


# ---------------------------
# 波动率
# ---------------------------
def get_30d_vol() -> float:
    df = get_klines("1d", 60)
    if len(df) < 30:
        return float("nan")

    rets = df["close"].pct_change().dropna()
    if len(rets) < 30:
        return float("nan")

    daily_std = rets.tail(30).std()
    return float(daily_std * math.sqrt(365))


# ---------------------------
# Funding / OI / mark price
# ---------------------------
def _get_hl_perp_ctx() -> dict:
    """获取 Hyperliquid 上 BTC perp 的资产上下文，带简单缓存。"""
    global _HL_PERP_CTX
    if _HL_PERP_CTX is not None:
        return _HL_PERP_CTX

    data = http_post_json(HL_INFO, {"type": "metaAndAssetCtxs"})
    if not data or len(data) < 2:
        raise RuntimeError("Hyperliquid metaAndAssetCtxs 返回异常")

    universe = data[0].get("universe") or []
    ctxs = data[1]
    idx = None
    for i, meta in enumerate(universe):
        if meta.get("name") == "BTC":
            idx = i
            break
    if idx is None:
        raise RuntimeError("Hyperliquid 中未找到 BTC 合约")
    if idx >= len(ctxs):
        raise RuntimeError("Hyperliquid 资产上下文索引越界")

    _HL_PERP_CTX = ctxs[idx]
    return _HL_PERP_CTX


def _get_funding_from_binance() -> float:
    if SIMULATE_BINANCE_FAIL:
        raise RuntimeError("SIMULATE_BINANCE_FAIL=1，模拟 Binance 失败")

    data = http_get(f"{BINANCE_FUT}/fapi/v1/fundingRate", {
        "symbol": "BTCUSDT",
        "limit": 1,
    })
    if not data:
        raise RuntimeError("Binance funding 返回空数据")
    return float(data[0]["fundingRate"])


def _get_funding_from_hyperliquid() -> float:
    ctx = _get_hl_perp_ctx()
    f = ctx.get("funding")
    if f is None:
        raise RuntimeError("Hyperliquid funding 字段缺失")
    return _normalize_hl_funding(float(f))


def _normalize_hl_funding(f_val: float) -> float:
    """将 Hyperliquid 的“每秒资金费率”对齐为 8 小时费率的量级。

    其他所有使用 funding 的逻辑（风险打分 / 通知输出等）都假设是 8h 费率。
    为避免出现 -0.00000 这样的近零展示或风险权重失真，这里统一处理。
    """
    # Hyperliquid 返回的 funding 为“每秒资金费率”，数量级远小于 Binance 的 8 小时费率。
    # 为了与 Binance 输出保持一致，做一个温和的单位对齐：
    # - 如果原值非常接近 0（<1e-5），按“每秒 * 8 小时”的尺度放大；
    # - 若未来官方直接返回 8h 费率或数值本身已正常，则保持原样避免过度放大。
    if abs(f_val) < 1e-5:
        f_val *= 8 * 60 * 60
    return f_val


def get_funding() -> float:
    """统一 funding 指标获取，支持多数据源 + 回退。"""
    primary = (DERIV_SOURCE or "hyperliquid").lower()
    if primary == "binance":
        order = ["binance", "hyperliquid"]
    else:
        order = ["hyperliquid", "binance"]

    for src_name in order:
        try:
            if src_name == "binance":
                DATA_SOURCES["funding"] = "binance"
                return _get_funding_from_binance()
            elif src_name == "hyperliquid":
                DATA_SOURCES["funding"] = "hyperliquid"
                return _get_funding_from_hyperliquid()
        except Exception as e:
            print(f"⚠️ get_funding 使用 {src_name} 失败: {e}")

    return float("nan")


def _get_oi_from_binance() -> float:
    if SIMULATE_BINANCE_FAIL:
        raise RuntimeError("SIMULATE_BINANCE_FAIL=1，模拟 Binance 失败")

    oi = http_get(f"{BINANCE_FUT}/fapi/v1/openInterest", {"symbol": "BTCUSDT"})
    price = http_get(f"{BINANCE_FUT}/fapi/v1/ticker/price", {"symbol": "BTCUSDT"})
    if not oi or not price:
        raise RuntimeError("Binance OI / 价格返回空")
    return float(oi["openInterest"]) * float(price["price"])


def _get_oi_from_hyperliquid() -> float:
    ctx = _get_hl_perp_ctx()
    oi = ctx.get("openInterest")
    px = ctx.get("markPx")
    if oi is None or px is None:
        raise RuntimeError("Hyperliquid openInterest 或 markPx 字段缺失")
    return float(oi) * float(px)


def get_open_interest() -> float:
    """统一 open interest 指标获取，支持多数据源 + 回退。"""
    primary = (DERIV_SOURCE or "hyperliquid").lower()
    if primary == "binance":
        order = ["binance", "hyperliquid"]
    else:
        order = ["hyperliquid", "binance"]

    for src_name in order:
        try:
            if src_name == "binance":
                DATA_SOURCES["oi"] = "binance"
                return _get_oi_from_binance()
            elif src_name == "hyperliquid":
                DATA_SOURCES["oi"] = "hyperliquid"
                return _get_oi_from_hyperliquid()
        except Exception as e:
            print(f"⚠️ get_open_interest 使用 {src_name} 失败: {e}")

    return float("nan")


def get_mark_price() -> float:
    """统一获取 BTC 的 mark price（优先 Hyperliquid Perp，备用 Binance 永续）。"""
    # 优先：使用 Hyperliquid BTC perp 的 markPx
    try:
        ctx = _get_hl_perp_ctx()
        px = ctx.get("markPx") if ctx else None
        if px is not None:
            DATA_SOURCES["mark_price"] = "hyperliquid"
            return float(px)
    except Exception as e:
        print(f"⚠️ get_mark_price 使用 hyperliquid 失败: {e}")

    # 回退：使用 Binance U 本位永续的 premiumIndex 作为 mark price
    try:
        if SIMULATE_BINANCE_FAIL:
            raise RuntimeError("SIMULATE_BINANCE_FAIL=1，模拟 Binance 失败")

        data = http_get(f"{BINANCE_FUT}/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})
        if data and isinstance(data, dict) and "markPrice" in data:
            DATA_SOURCES["mark_price"] = "binance"
            return float(data["markPrice"])
    except Exception as e:
        print(f"⚠️ get_mark_price 使用 binance 失败: {e}")

    return float("nan")

# ---------------------------
# OI History for Risk Factor
# ---------------------------
def _load_oi_cache() -> list:
    """从本地 JSON 加载 HL OI 历史缓存。"""
    if not os.path.exists(OI_CACHE_FILE):
        return []
    try:
        with open(OI_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 读取 {OI_CACHE_FILE} 失败: {e}")
        return []


def _save_oi_cache(data: list):
    """保存 HL OI 历史缓存到 JSON。"""
    try:
        with open(OI_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"⚠️ 写入 {OI_CACHE_FILE} 失败: {e}")


def record_hl_oi_for_cache():
    """
    每次运行脚本时，尝试从 Hyperliquid 读取当前 OI（名义价值），
    并写入本地缓存 JSON，作为 Binance OI 历史失败时的备份。
    """
    try:
        ctx = _get_hl_perp_ctx()
        oi = ctx.get("openInterest")
        px = ctx.get("markPx")
        if oi is None or px is None:
            raise RuntimeError("Hyperliquid openInterest 或 markPx 字段缺失")
        notional = float(oi) * float(px)
    except Exception as e:
        print(f"⚠️ 记录 HL OI 缓存失败: {e}")
        return

    data = _load_oi_cache()
    now_ts = datetime.now(timezone.utc).timestamp()
    data.append({"ts": now_ts, "oi": notional})

    # 只保留最近 120 条（约 4 个月），避免文件无限增长
    if len(data) > 120:
        data = data[-120:]

    _save_oi_cache(data)


def _load_hl_oi_history_from_cache(days: int = 7) -> list:
    data = _load_oi_cache()
    if not data:
        return []

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # 按日期去重：每天只保留最新的一条
    daily = {}
    for item in data:
        try:
            ts = float(item.get("ts", 0))
        except Exception:
            print(f"⚠️ 跳过无效缓存记录（ts 无法解析）：{item}")
            continue
        dt = datetime.fromtimestamp(ts, timezone.utc)
        if dt < cutoff:
            continue
        day_key = dt.date()
        try:
            oi_val = float(item.get("oi"))
        except Exception:
            print(f"⚠️ 跳过无效缓存记录（oi 无法解析）：{item}")
            continue

        # 覆盖写入：保留每天最新的一条（时间戳更大的覆盖旧的）
        prev = daily.get(day_key)
        if not prev or ts >= float(prev.get("ts", 0)):
            daily[day_key] = {"ts": ts, "oi": oi_val}

    # 按日期排序
    sorted_items = [daily[k] for k in sorted(daily.keys())]

    return [x["oi"] for x in sorted_items]



def get_oi_history_from_binance(days: int = 7) -> list:
    """
    从 Binance U 本位永续获取 OI 日线历史（名义价值）。
    优先用于计算 OI 趋势，失败时再使用 HL 缓存。
    """
    if SIMULATE_BINANCE_FAIL:
        raise RuntimeError("SIMULATE_BINANCE_FAIL=1，模拟 Binance 失败")

    # Binance futures OI 历史接口：openInterestHist
    data = http_get(f"{BINANCE_FUT}/futures/data/openInterestHist", {
        "symbol": "BTCUSDT",
        "period": "1d",
        "limit": max(days + 1, 10),
    })
    if not data:
        raise RuntimeError("Binance OI history 返回空数据")

    # 按时间从旧到新排序（接口通常已按时间升序，但这里防御性处理一次）
    try:
        data_sorted = sorted(data, key=lambda x: int(x.get("timestamp", 0)))
    except Exception:
        data_sorted = data

    # sumOpenInterest 是以 BTC 计价的持仓数量；需要乘以价格才是名义价值。
    # 这里为了简化 & 避免额外价格请求，我们直接用 sumOpenInterest 做“相对变化”的衡量即可。
    # 对于趋势而言，使用 BTC 数量和使用名义价值的方向是一致的。
    oi_list = [float(x["sumOpenInterest"]) for x in data_sorted]

    # 只取最后 N 天
    if len(oi_list) >= days:
        oi_list = oi_list[-days:]

    return oi_list


def compute_oi_trend(days: int = 7) -> float:
    """
    统一的 OI 趋势计算：
    1. 优先使用 Binance 的 OI 历史（sumOpenInterest）计算 7 日斜率；
    2. 如果失败 / 数据不足，则回退到本地缓存的 HL OI；
    3. 如果仍然不足，则返回 0（视为中性）。
    """
    oi_list = []
    # 1) 优先 Binance
    try:
        oi_list = get_oi_history_from_binance(days)
        if len(oi_list) >= 3:
            DATA_SOURCES["oi_history"] = "binance"
    except Exception as e:
        print(f"⚠️ compute_oi_trend 使用 Binance OI history 失败: {e}")
        oi_list = []

    # 2) 回退：HL 本地缓存
    if len(oi_list) < 3:
        oi_list = _load_hl_oi_history_from_cache(days)
        if len(oi_list) >= 3:
            DATA_SOURCES["oi_history"] = "hyperliquid_cache"

    if len(oi_list) < 3:
        # 数据太少，无法算趋势 → 当作中性，但仍记录数据来源方便排查
        if oi_list and not DATA_SOURCES.get("oi_history"):
            DATA_SOURCES["oi_history"] = "hyperliquid_cache"
        if not DATA_SOURCES.get("oi_history"):
            DATA_SOURCES["oi_history"] = "unavailable"
        return 0.0

    oi_vals = np.array(oi_list, dtype=float)
    t = np.arange(len(oi_vals))

    # 最小二乘线性回归 slope
    slope = np.polyfit(t, oi_vals, 1)[0]

    # 标准化为“相对每日增长率”
    norm_slope = slope / np.mean(oi_vals)

    return float(norm_slope)


def oi_risk_from_trend(tr: float) -> float:
    """根据 7 日 OI 趋势确定风险预算因子（tr 是归一化后的斜率，比如 7 日涨幅）"""
    if tr <= 0:
        return 1.0     # 杠杆在撤，风险降低
    elif tr < 0.01:
        return 0.9     # 略微上升
    elif tr < 0.03:
        return 0.7     # 中等风险
    elif tr < 0.07:
        return 0.4     # 杠杆明显上升
    else:
        return 0.1     # 杠杆过度堆积（危险）


def funding_risk_from_funding(f: float) -> float:
    """
    根据 funding rate 估算风险：
    - 先按 |f| 的大小判断“拥挤程度”
    - 再按方向（正/负）略微调整：正 funding 对多头更危险
    f 是每 8 小时或每天的 funding rate，比如 0.0001 = 0.01%
    """

    if math.isnan(f):
        return 1.0

    af = abs(f)

    # 1) 拥挤程度：用绝对值切桶
    if af < 0.00005:
        base = 1.0   # 非常接近 0，市场相对均衡
    elif af < 0.0002:
        base = 0.8   # 有些拥挤，但还在可接受范围
    elif af < 0.0008:
        base = 0.5   # 比较极端的情绪了
    else:
        base = 0.2   # 极端拥挤，容易发生挤兑/挤爆仓行情

    # 2) 按方向微调
    if f > 0:
        # 多头付钱给空头：做多更危险 → 稍微再降一点风险得分
        funding_risk = max(0.1, base - 0.1)
    elif f < 0:
        # 空头付钱给多头：对现货/多头来说风险略低一点
        funding_risk = min(1.0, base + 0.1)
    else:
        funding_risk = base

    return funding_risk


# ---------------------------
# SSR-like
# ---------------------------
STABLES = [
    "tether", "usd-coin", "dai", "first-digital-usd",
    "frax", "ethena-usde", "usdd", "paypal-usd"
]


def get_ssr_like() -> float:
    btc = http_get(f"{CG}/coins/bitcoin", {
        "localization": "false", "tickers": "false", "market_data": "true"
    })
    if not btc:
        return float("nan")

    btc_mcap = btc["market_data"]["market_cap"]["usd"]

    ids = ",".join(STABLES)
    st = http_get(f"{CG}/coins/markets", {
        "vs_currency": "usd",
        "ids": ids,
        "order": "market_cap_desc",
        "per_page": len(STABLES),
        "page": 1,
        "sparkline": "false",
    })
    DATA_SOURCES["ssr"] = "coingecko" 
    if not st:
        return float("nan")

    stable_mcap = sum(x.get("market_cap", 0.0) for x in st)
    if stable_mcap == 0:
        return float("nan")

    return float(btc_mcap / stable_mcap)


# ---------------------------
# scoring
# ---------------------------
@dataclass
class Snapshot:
    price: float
    mark_price: float
    ma200d: float
    mayer: float
    ma200w: float
    dist200w: float
    ssr: float
    vol30d: float
    funding: float  # 统一用 8 小时费率的量级（Hyperliquid 会在获取时归一化）
    oi: float
    trend7d: float  # 过去 7 天涨跌幅（相对值，例如 0.05=+5%）


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )


def wilder_atr(tr_list: list[float], window: int) -> float:
    if not tr_list:
        return float("nan")

    if len(tr_list) < window:
        return float(sum(tr_list) / len(tr_list))

    atr = sum(tr_list[:window]) / window
    for tr in tr_list[window:]:
        atr = (atr * (window - 1) + tr) / window
    return float(atr)


def compute_adaptive_window(atr14: float, atr60: float, base_window: int = 30,
                            min_window: int = 10, max_window: int = 45) -> int:
    ratio = atr60 / max(atr14, 1e-9)
    window_raw = round(base_window * ratio)
    window = min(max(window_raw, min_window), max_window)
    return int(window)


def compute_volatility_context(df: pd.DataFrame, base_window: int = 30) -> dict:
    if df is None or df.empty:
        return {
            "ma": float("nan"),
            "atr": float("nan"),
            "prev_close": float("nan"),
            "atr14": float("nan"),
            "atr60": float("nan"),
            "window": base_window,
        }

    closes = df["close"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()

    if len(closes) == 0:
        return {
            "ma": float("nan"),
            "atr": float("nan"),
            "prev_close": float("nan"),
            "atr14": float("nan"),
            "atr60": float("nan"),
            "window": base_window,
        }

    prev_closes = np.concatenate(([closes[0]], closes[:-1]))
    tr_values = [true_range(h, l, pc) for h, l, pc in zip(highs, lows, prev_closes)]

    atr14 = wilder_atr(tr_values, 14)
    atr60 = wilder_atr(tr_values, 60)
    window = compute_adaptive_window(atr14, atr60, base_window=base_window)

    ma = float(np.mean(closes[-window:]))
    atr = wilder_atr(tr_values, window)
    prev_close = float(closes[-2]) if len(closes) >= 2 else float(closes[-1])

    return {
        "ma": ma,
        "atr": atr,
        "prev_close": prev_close,
        "atr14": atr14,
        "atr60": atr60,
        "window": window,
    }


def score_valuation(mayer, dist):
    score = 0
    if not math.isnan(mayer):
        if mayer < 0.8: score += 3
        elif mayer < 1.2: score += 2
        elif mayer < 2.4: score += 0
        else: score -= 3

    if not math.isnan(dist):
        if dist < -0.10: score += 3
        elif dist < 0.30: score += 1
        elif dist < 1.00: score += 0
        else: score -= 2
    return score


def score_liquidity(ssr):
    if math.isnan(ssr): return 0
    if ssr < 1.5: return 2
    if ssr < 3.0: return 1
    if ssr < 5.0: return 0
    return -1


def get_7d_trend() -> float:
    """
    计算过去 7 天的涨跌幅：
    trend = (最新收盘价 / 7 天前收盘价) - 1
    """
    df = get_klines("1d", 10)
    if len(df) < 8:
        return float("nan")

    price_now = df["close"].iloc[-1]
    # price_1d_ago = df["close"].iloc[-2]
    price_7d_ago = df["close"].iloc[-8]  # 7 天前的收盘
    if price_7d_ago <= 0:
        return float("nan")

    return float(price_now / price_7d_ago - 1.0)


def safety_from_vol(vol30: float) -> float:
    if math.isnan(vol30):
        return 0.6

    if vol30 < 0.4:
        return 1.0
    if vol30 < 0.7:
        return 0.7
    if vol30 < 1.0:
        return 0.4
    return 0.1


def safety_from_oi_trend(oi_tr: float) -> float:
    if math.isnan(oi_tr):
        return 0.6

    if oi_tr < 0.0:
        return 1.0
    if oi_tr < 0.03:
        return 0.8
    if oi_tr < 0.07:
        return 0.5
    return 0.2


def safety_from_funding(funding: float) -> float:
    if math.isnan(funding):
        return 0.6

    if funding < 0.0:
        return 1.0
    if funding < 0.01:
        return 0.8
    if funding < 0.03:
        return 0.5
    return 0.2


def compute_safety_score(snapshot: Snapshot, oi_tr: float, alpha: float = 0.25) -> float:
    s_vol = safety_from_vol(snapshot.vol30d)
    s_oi = safety_from_oi_trend(oi_tr)
    s_funding = safety_from_funding(snapshot.funding)

    factors = [s_vol, s_oi, s_funding]
    weakest = min(factors)
    avg = sum(factors) / len(factors)
    safety = weakest + alpha * (avg - weakest)
    return max(0.0, min(1.0, safety))


# ---------------------------
# 决策
# ---------------------------
def compute_risk_brake(snapshot: Snapshot, oi_tr: float, max_brake: float = 0.6) -> float:
    safety = compute_safety_score(snapshot, oi_tr)
    base_brake = -max_brake * (1.0 - safety)

    price_tr = snapshot.trend7d
    if math.isnan(price_tr):
        return base_brake * 0.7

    if price_tr <= -0.05:
        return base_brake * 0.3
    if price_tr <= 0:
        return base_brake * 0.5

    return base_brake


def explain_risk_brake(snapshot: Snapshot, risk_brake: float, oi_tr: float) -> str:
    reasons = []
    price_tr = snapshot.trend7d

    s_vol = safety_from_vol(snapshot.vol30d)
    s_oi = safety_from_oi_trend(oi_tr)
    s_funding = safety_from_funding(snapshot.funding)
    safety = compute_safety_score(snapshot, oi_tr)

    reasons.append(
        f"安全分综合为 {safety:.2f}（波动 {s_vol:.2f} / OI 趋势 {s_oi:.2f} / funding {s_funding:.2f}）。"
    )

    if math.isnan(price_tr):
        reasons.append("价格 7 日趋势缺失，刹车力度略微削弱以避免过度保守。")
    elif price_tr <= -0.05:
        reasons.append("过去 7 日显著下跌，刹车力度大幅折扣以优先抓取超跌机会。")
    elif price_tr <= 0:
        reasons.append("价格 7 日趋势不强，刹车力度折半以留给下跌机会空间。")

    if risk_brake == 0:
        reasons.append("整体结构较安全，未踩刹车。")
    elif risk_brake > -0.2:
        reasons.append("轻微收缩仓位，避免在拥挤结构中加速建仓。")
    else:
        reasons.append("杠杆或 funding 偏热，显著下调倍数以防 FOMO 风险。")

    return "；".join(reasons)


def compute_volatility_edge(snapshot: Snapshot, ma: float, atr: float, prev_close: float):
    close = snapshot.price
    if math.isnan(ma) or math.isnan(atr) or math.isnan(prev_close) or prev_close == 0:
        return 0.0, float("nan"), float("nan")

    daily_ret = (close - prev_close) / prev_close
    z = (close - ma) / max(atr, 1e-9)

    edge = 0.0
    if z <= -1.5:
        edge = 0.6
    elif z <= -1.0:
        edge = 0.3

    if daily_ret > 0.04:
        edge = 0.0

    return edge, z, daily_ret


def explain_volatility_edge(edge: float, z: float, daily_ret: float) -> str:
    if math.isnan(z) or math.isnan(daily_ret):
        return "波动数据不足，未启用超跌加成。"

    if daily_ret > 0.04:
        return "当日已大幅反弹，波动套利层不加成。"

    if edge == 0:
        return "价格偏离均值不够极端，未触发超跌加仓。"

    if edge >= 0.6:
        return f"价格显著低于均值（z={z:.2f}），触发强力超跌加仓。"

    return f"价格明显低于均值（z={z:.2f}），触发温和超跌加仓。"


def decide(snapshot: Snapshot):
    val = score_valuation(snapshot.mayer, snapshot.dist200w)
    liq = score_liquidity(snapshot.ssr)
    total = val + liq

    if total >= 9:
        m = 10
        txt = "极端低估 + 极低风险，大底建仓（10x）"

    elif total >= 7:
        m = 6
        txt = "显著低估，强力加仓（6x）"

    elif total >= 4:
        m = 4
        txt = "明显低估，积极建仓（4x）"

    elif total >= 2:
        m = 2.5
        txt = "轻度低估，主动建仓（2.5x）"

    elif total >= 0:
        m = 1.5
        txt = "中性区间，正常定投（1.5x）"

    elif total >= -2:
        m = 0.75
        txt = "偏高估，减少投入（0.75x）"

    else:
        m = 0
        txt = "高估 + 高风险，暂停定投（0x）"

    return m, txt, total

def build_risk_hint(snap: Snapshot) -> str:
    """
    根据 funding / OI 给出一句自然语言风险提示。
    阈值是经验参数，可以日后根据感觉调。
    """
    hints = []

    # Funding 过热 / 过冷
    if not math.isnan(snap.funding):
        if snap.funding > 0.0008:
            hints.append("资金费率明显偏正，多头较拥挤，短期回撤风险增加。")
        elif snap.funding < -0.0005:
            hints.append("资金费率明显为负，空头较拥挤，存在空头挤压的可能。")

    # OI 绝对值很大（这里用一个比较保守的阈值）
    if not math.isnan(snap.oi):
        # snap.oi 是名义价值（USDT），这里粗暴用 20B / 30B 做分段
        if snap.oi > 30_000_000_000:
            hints.append("永续合约 OI 名义价值处于极高水平，杠杆堆积，波动可能被明显放大。")
        elif snap.oi > 20_000_000_000:
            hints.append("永续合约 OI 处于偏高区间，需关注杠杆集中带来的波动放大风险。")

    if not hints:
        return "风险水平中性，暂未出现资金费率或杠杆明显极端的信号。"

    return "；".join(hints)


# ---------------------------
# 主流程封装：给通知 / 交易用的接口
# ---------------------------
def run_today(base: float = 30.0):
    """
    计算当日所有指标 + 打分 + 建议。
    返回：
        snapshot: Snapshot 实例（全量指标）
        mult: 估值倍数（估值 + 流动性）
        text: 文字说明
        score: 综合得分
        base: 基础定投金额
        invest: 建议实际投入金额（base * final_mult）
        risk_hint: 基于 funding / OI 的一句风险提示
        final_mult: 综合估值 + 风险刹车 + 波动套利后的最终倍数
    """
    # 现货收盘价（来自 K 线）+ 衍生品 mark price（更加稳定）
    price, ma200d, mayer = get_mayer()
    mark_price = get_mark_price()

    ma200w, dist = get_200w_distance(price)
    ssr = get_ssr_like()
    vol = get_30d_vol()
    funding = get_funding()
    oi = get_open_interest()
    trend7d = get_7d_trend()

    snap = Snapshot(
        price=price,
        mark_price=mark_price,
        ma200d=ma200d,
        mayer=mayer,
        ma200w=ma200w,
        dist200w=dist,
        ssr=ssr,
        vol30d=vol,
        funding=funding,
        oi=oi,
        trend7d=trend7d,
    )

    # 先记录一份当日 HL OI 到本地缓存（如果 HL 可用）
    record_hl_oi_for_cache()
    # 计算 OI 7 日趋势（优先 Binance，失败则用 HL 缓存）
    oi_tr = compute_oi_trend(days=7)
    kline_df = get_klines("1d", 200)
    vol_ctx = compute_volatility_context(kline_df, base_window=30)

    mult, text, score = decide(snap)
    risk_brake = compute_risk_brake(snap, oi_tr)
    volatility_edge, z, daily_ret = compute_volatility_edge(
        snap, vol_ctx["ma"], vol_ctx["atr"], vol_ctx["prev_close"],
    )

    raw_final_mult = mult + risk_brake + volatility_edge
    final_mult = min(max(raw_final_mult, 0.0), 10.0)
    invest = base * final_mult
    risk_hint = build_risk_hint(snap)

    risk_brake_text = explain_risk_brake(snap, risk_brake, oi_tr)
    vol_edge_text = explain_volatility_edge(volatility_edge, z, daily_ret)

    return {
        "snapshot": snap,
        "mult": mult,
        "text": text,
        "score": score,
        "base": base,
        "invest": invest,
        "risk_hint": risk_hint,
        "risk_brake": risk_brake,
        "risk_brake_text": risk_brake_text,
        "volatility_edge": volatility_edge,
        "volatility_edge_text": vol_edge_text,
        "final_mult": final_mult,
        "z_score": z,
        "daily_ret": daily_ret,
        "atr14": vol_ctx["atr14"],
        "atr60": vol_ctx["atr60"],
        "adaptive_window": vol_ctx["window"],
        "sources": DATA_SOURCES.copy(),
    }



# ---------------------------
# CLI 打印版（本地跑用）
# ---------------------------
def main():
    print("Fetching BTC metrics...")
    result = run_today(base=30.0)

    snap = result["snapshot"]
    mult = result["mult"]
    final_mult = result["final_mult"]
    risk_brake = result["risk_brake"]
    volatility_edge = result["volatility_edge"]
    text = result["text"]
    score = result["score"]
    invest = result["invest"]
    base = result["base"]

    print("\n===== 今日 BTC 定投决策 =====")
    print(f"价格: {snap.price:,.2f}")
    print(f"Mark Price（衍生品）: {snap.mark_price:,.2f}")
    print(f"过去 7 天涨跌幅: {snap.trend7d*100:.2f}%")   
    print(f"Mayer Multiple: {snap.mayer:.3f}")
    print(f"距200W: {snap.dist200w*100:.2f}%")
    print(f"SSR-like: {snap.ssr:.3f}")
    print(f"30D 年化波动率: {snap.vol30d:.3f}")
    print(f"Funding Rate: {snap.funding:.5f}")
    print(f"Open Interest 名义价值: {snap.oi:,.0f}")
    print(f"综合得分: {score}")

    print(f"\n基础定投金额：{base:.2f} USDT")
    print(f"估值倍数：{mult}x")
    print(f"风险刹车：{risk_brake:+.2f}")
    print(f"波动套利加成：{volatility_edge:+.2f}")
    print(f"最终建议倍数：{final_mult}x")
    print(f"今日实际应投入：{invest:.2f} USDT")
    print(f"定投倍数解释：{text}")
    print(f"风险刹车解释: {result.get('risk_brake_text', '')}")
    print(f"波动套利解释: {result.get('volatility_edge_text', '')}")

    # print("\n--- 数据源说明 ---")
    # print("K线 / 价格：优先 Binance 现货 BTCUSDT，失败时回退 OKX 现货 BTC-USDT。")
    # print("Mark Price / Funding / OI（杠杆、情绪）：优先 Hyperliquid BTC 永续，失败时回退 Binance U 本位永续 BTCUSDT。")
    # print("SSR-like：CoinGecko 上 BTC 市值 ÷ 稳定币篮子（USDT / USDC / DAI / FDUSD / FRAX / USDe / USDD / PYUSD）。")
    print("\n--- 本次运行实际数据源 ---")
    for k, v in DATA_SOURCES.items():
        if v: print(f"{k}: {v}")



if __name__ == "__main__":
    main()
