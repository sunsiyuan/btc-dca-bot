#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
from dca_btc import run_today, Snapshot  # Snapshot 主要是类型提示，可选
from dotenv import load_dotenv
load_dotenv()


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FT_SENDKEY = os.getenv("FT_SENDKEY")
WEBHOOK_URLS = os.getenv("WEBHOOK_URLS", "")  # 可选：多渠道 HTTP webhook
BASE_AMOUNT = float(os.getenv("BASE_DCA_USDT", "30"))  # 可以用 env 覆盖 base

# ===== 微信推送（Server酱） =====
def send_wechat_by_ft(content: str):
    """
    使用方糖/Server酱推送微信消息。
    遇到错误自动跳过，不影响主程序。
    """
    if not FT_SENDKEY:
        print("方糖未配置，跳过通知")
        return
    try:
        url = f"https://sctapi.ftqq.com/{FT_SENDKEY}.send"
        data = {
            "title": "BTC定投今日推送",
            "desp": content,
        }
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[WARN] 微信（Server酱）推送失败: {e}")
        return False

def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram 未配置，跳过通知")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print("发送 Telegram 失败：", e)


def send_webhooks(text: str):
    """
    多渠道 webhook，适合 Slack / Discord / 自己的通知系统。
    WEBHOOK_URLS 支持多个，用逗号分隔。
    """
    if not WEBHOOK_URLS:
        return
    for raw in WEBHOOK_URLS.split(","):
        url = raw.strip()
        if not url:
            continue
        try:
            r = requests.post(url, json={"text": text}, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"发送 Webhook ({url}) 失败：", e)


def build_message(result: dict) -> str:
    snap: Snapshot = result["snapshot"]
    mult = result["mult"]
    text = result["text"]
    score = result["score"]
    base = result["base"]
    invest = result["invest"]
    risk_hint = result.get("risk_hint", "风险水平中性，暂无明显极端信号。")

    lines = [
        "📊 今日 BTC 定投模型结果",
        "",
        f"📈 现货收盘价（close）: {snap.price:,.2f} USDT",
        f"💹 Mark Price（衍生品）: {snap.mark_price:,.2f} USDT",
        "",
        f"🔵 Mayer Multiple: {snap.mayer:.3f}",
        f"🟣 距 200W MA: {snap.dist200w*100:.2f}%",
        f"🟢 SSR-like: {snap.ssr:.3f}",
        f"🟠 30D 年化波动率: {snap.vol30d:.3f}",
        f"📉 过去 7 天涨跌幅: {snap.trend7d*100:.2f}%",
        f"🔺 Funding Rate: {snap.funding:.5f}",
        f"📦 Open Interest 名义价值: {snap.oi:,.0f}",
        "",
        f"🧮 综合得分: {score}",
        "",
        f"💰 基础定投金额: {base:.2f} USDT",
        f"📌 建议定投倍数: {mult}x",
        f"👉 今日建议投入: {invest:.2f} USDT",
        "",
        f"⚠️ 风险提示: {risk_hint}",
        "",
        f"📘 说明: {text}",
    ]
    return "\n".join(lines)



def main():
    # 可以用环境变量覆盖基础金额
    result = run_today(base=BASE_AMOUNT)
    msg = build_message(result)
    print(msg)
    send_telegram(msg)
    send_wechat_by_ft(msg)
    send_webhooks(msg)


if __name__ == "__main__":
    main()
