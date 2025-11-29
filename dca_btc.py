#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BTC 定投黄金模型 —— TODAY 版本（无历史回测）
--------------------------------------------
数据源：
- Binance 现货 & 合约 API（免费）
- CoinGecko（免费）

功能：
- 计算 Mayer Multiple（200D）
- 相对 200W MA
- SSR-like（BTC市值 / 稳定币篮子）
- 30D 年化波动率
- Funding rate
- Open Interest 名义价值

输出：
- 指标快照
- 定投倍数
- 总结性建议
"""

import math
import requests
import pandas as pd
from dataclasses import dataclass

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUT = "https://fapi.binance.com"
CG = "https://api.coingecko.com/api/v3"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "btc-dca-today/1.1"})


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


# ---------------------------
# BTC K线
# ---------------------------
def get_klines(interval="1d", limit=500) -> pd.DataFrame:
    data = http_get(f"{BINANCE_SPOT}/api/v3/klines", {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": limit
    })
    if not data:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    df.set_index("open_time", inplace=True)
    return df


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

    closes = df["close"]
    ma200w = closes.tail(200).mean()
    if price is None:
        price = closes.iloc[-1]

    dist = (price - ma200w) / ma200w
    return float(ma200w), float(dist)


# ---------------------------
# 30D 年化波动率
# ---------------------------
def get_30d_vol() -> float:
    df = get_klines("1d", 80)
    if df.empty:
        return float("nan")

    prices = df["close"]
    rets = (prices / prices.shift(1)).dropna().apply(math.log)
    if len(rets) < 15:
        return float("nan")

    daily_std = rets.tail(30).std()
    return float(daily_std * math.sqrt(365))


# ---------------------------
# Funding / OI
# ---------------------------
def get_funding() -> float:
    data = http_get(f"{BINANCE_FUT}/fapi/v1/fundingRate", {
        "symbol": "BTCUSDT",
        "limit": 1
    })
    if not data:
        return float("nan")
    return float(data[0]["fundingRate"])


def get_open_interest() -> float:
    oi = http_get(f"{BINANCE_FUT}/fapi/v1/openInterest", {"symbol": "BTCUSDT"})
    price = http_get(f"{BINANCE_FUT}/fapi/v1/ticker/price", {"symbol": "BTCUSDT"})
    if not oi or not price:
        return float("nan")
    return float(oi["openInterest"]) * float(price["price"])


# ---------------------------
# SSR-like
# ---------------------------
STABLES = [
    "tether", "usd-coin", "dai", "first-digital-usd",
    "frax", "ethena-usde", "usdd", "paypal-usd"
]


def get_ssr_like() -> float:
    btc = http_get(f"{CG}/coins/bitcoin", {
        "localization":"false","tickers":"false","market_data":"true"
    })
    if not btc:
        return float("nan")

    btc_mc = btc["market_data"]["market_cap"]["usd"]

    ids = ",".join(STABLES)
    data = http_get(f"{CG}/coins/markets", {
        "vs_currency": "usd", "ids":ids, "per_page":len(STABLES)
    })
    if not data:
        return float("nan")

    stable_mc = sum([c.get("market_cap") or 0 for c in data])

    if stable_mc <= 0:
        return float("nan")

    return float(btc_mc / stable_mc)


# ---------------------------
# scoring
# ---------------------------
@dataclass
class Snapshot:
    price: float
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
        if funding <= -0.0002: score += 1  # # 明确偏空 → 加分
        elif funding < 0.0002: score += 0  # 轻微波动（微正/微负）→ 不加不减
        else: score -= 2 # 明确偏热 → 扣分
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
    price, ma200d, mayer = get_mayer()
    ma200w, dist = get_200w_distance(price)
    ssr = get_ssr_like()
    vol = get_30d_vol()
    funding = get_funding()
    oi = get_open_interest()

    snap = Snapshot(price, ma200d, mayer, ma200w, dist, ssr, vol, funding, oi)
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
