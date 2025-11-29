#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BTC 定投黄金模型 —— TODAY 版本（无历史回测）
--------------------------------------------
数据源：
- Binance 现货 & 合约 API（免费）
- Hyperliquid Perp Info API（免费）
- OKX 现货 BTC-USDT（备用 K 线数据源，Binance 失败时回退）
- CoinGecko（仅 SSR-like 指标）

功能：
- 计算 Mayer Multiple（200D）
- 相对 200W MA
- SSR-like（BTC市值 / 稳定币篮子）
- 30D 年化波动率
- Funding rate
- Open Interest 名义价值
- 综合评分 + 建议定投倍数

输出：
- 指标快照
- 定投倍数
- 总结性建议
"""

import math
import os
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
from dataclasses import dataclass

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUT = "https://fapi.binance.com"
OKX_SPOT = "https://www.okx.com"
CG = "https://api.coingecko.com/api/v3"
HL_INFO = "https://api.hyperliquid.xyz/info"

# 数据源配置（支持多数据源 + 回退）
DATA_SOURCE = os.getenv("DATA_SOURCE", "binance").lower()  # "binance" 或 "okx"
DERIV_SOURCE = os.getenv("DERIV_SOURCE", "hyperliquid").lower()  # "binance" 或 "hyperliquid"
SIMULATE_BINANCE_FAIL = os.getenv("SIMULATE_BINANCE_FAIL", "0") == "1"

# Hyperliquid perp 资产上下文缓存（避免重复请求）
_HL_PERP_CTX = None

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-dca-today/1.2"})


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
                return _get_klines_from_binance(interval, limit)
            elif src_name == "okx":
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
    return float(f)


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
                return _get_funding_from_binance()
            elif src_name == "hyperliquid":
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
                return _get_oi_from_binance()
            elif src_name == "hyperliquid":
                return _get_oi_from_hyperliquid()
        except Exception as e:
            print(f"⚠️ get_open_interest 使用 {src_name} 失败: {e}")

    return float("nan")


def get_mark_price() -> float:
    """统一获取 BTC 的 mark price（优先 Binance 永续，备用 Hyperliquid）。"""
    # 优先使用 Binance U 本位永续的 premiumIndex 作为 mark price
    try:
        if SIMULATE_BINANCE_FAIL:
            raise RuntimeError("SIMULATE_BINANCE_FAIL=1，模拟 Binance 失败")

        data = http_get(f"{BINANCE_FUT}/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})
        if data and isinstance(data, dict) and "markPrice" in data:
            return float(data["markPrice"])
    except Exception as e:
        print(f"⚠️ get_mark_price 使用 binance 失败: {e}")

    # 回退：使用 Hyperliquid BTC perp 的 markPx
    try:
        ctx = _get_hl_perp_ctx()
        px = ctx.get("markPx") if ctx else None
        if px is not None:
            return float(px)
    except Exception as e:
        print(f"⚠️ get_mark_price 使用 hyperliquid 失败: {e}")

    return float("nan")


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
    funding: float
    oi: float


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


def score_risk(vol, funding):
    score = 0
    if not math.isnan(vol):
        if vol < 0.6: score += 1
        elif vol < 1.0: score += 0
        else: score -= 2

    if not math.isnan(funding):
        if funding <= -0.0002: score += 1  # 明确偏空 → 加分
        elif funding < 0.0002: score += 0  # 轻微波动（微正/微负）→ 不加不减
        else: score -= 2  # 明确偏热 → 扣分
    return score


# ---------------------------
# 决策
# ---------------------------
def decide(snapshot: Snapshot):
    val = score_valuation(snapshot.mayer, snapshot.dist200w)
    liq = score_liquidity(snapshot.ssr)
    risk = score_risk(snapshot.vol30d, snapshot.funding)
    total = val + liq + risk

    if total >= 7:
        m = 5
        txt = "估值极低+风险极低抄大底模式（5x）"
    elif total >= 4:
        m = 3
        txt = "明显低估，积极建仓（3x）。"
    elif total >= 2:
        m = 1.5
        txt = "估值略低，机会较好（1.5x）。"
    elif total >= 0:
        m = 1
        txt = "中性区间，正常定投（1x）。"
    elif total >= -1:
        m = 0.5
        txt = "偏贵或风险偏高，减半定投（0.5x）。"
    else:
        m = 0
        txt = "高估+高风险，暂停定投（0x）。"

    return m, txt, total


# ---------------------------
# 主流程封装：给通知 / 交易用的接口
# ---------------------------
def run_today(base: float = 30.0):
    """
    计算当日所有指标 + 打分 + 建议。
    返回：
        snapshot: Snapshot 实例（全量指标）
        mult: 建议定投倍数
        text: 文字说明
        score: 综合得分
        base: 基础定投金额
        invest: 建议实际投入金额（base * mult）
    """
    # 现货收盘价（来自 K 线）+ 衍生品 mark price（更加稳定）
    price, ma200d, mayer = get_mayer()
    mark_price = get_mark_price()

    ma200w, dist = get_200w_distance(price)
    ssr = get_ssr_like()
    vol = get_30d_vol()
    funding = get_funding()
    oi = get_open_interest()

    snap = Snapshot(price, mark_price, ma200d, mayer, ma200w, dist, ssr, vol, funding, oi)
    mult, text, score = decide(snap)
    invest = base * mult
    return {
        "snapshot": snap,
        "mult": mult,
        "text": text,
        "score": score,
        "base": base,
        "invest": invest,
    }


# ---------------------------
# CLI 打印版（本地跑用）
# ---------------------------
def main():
    print("Fetching BTC metrics...")
    result = run_today(base=30.0)

    snap = result["snapshot"]
    mult = result["mult"]
    text = result["text"]
    score = result["score"]
    invest = result["invest"]
    base = result["base"]

    print("\n===== 今日 BTC 定投决策 =====")
    print(f"价格: {snap.price:,.2f}")
    print(f"Mark Price（衍生品）: {snap.mark_price:,.2f}")
    print(f"Mayer Multiple: {snap.mayer:.3f}")
    print(f"距200W: {snap.dist200w*100:.2f}%")
    print(f"SSR-like: {snap.ssr:.3f}")
    print(f"30D 年化波动率: {snap.vol30d:.3f}")
    print(f"Funding Rate: {snap.funding:.5f}")
    print(f"Open Interest 名义价值: {snap.oi:,.0f}")
    print(f"综合得分: {score}")

    print(f"\n基础定投金额：{base:.2f} USDT")
    print(f"建议定投倍数：{mult}x")
    print(f"今日实际应投入：{invest:.2f} USDT")
    print(f"解释：{text}")


if __name__ == "__main__":
    main()
