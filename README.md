# BTC DCA Bot

## 模型概览（DTC / dca_btc.py）
- **核心理念**：Dynamic DCA（DTC）模型由估值层、风险刹车层、机会层叠加，最终买入金额 = 基础金额 × `final_mult`，通过 `clamp(mult + risk_brake + volatility_edge)` 控制上下限。
- **估值层 (`mult`)**：基于 Mayer Multiple、200 周均线距离、SSR 等慢变量给出阶梯倍数，不重复惩罚风险。【F:dca_btc.py†L4-L99】
- **风险层 (`risk_brake`)**：只做刹车，依据 Funding、7 日 OI 趋势、30 日波动率生成安全分数，趋势向下时刹车减弱，趋势向上且杠杆累积时刹车全额生效。【F:dca_btc.py†L31-L55】
- **机会层 (`volatility_edge`)**：自适应 ATR 窗口识别快速超跌，Z 分数越低加成越大，若当日反抽则取消加仓，避免追高。【F:dca_btc.py†L56-L79】
- **设计哲学**：估值定仓、风险踩刹车、只在错杀日加仓，不做预测而是自适应结构。【F:dca_btc.py†L100-L107】

## 使用 dca_btc.py
- 本地运行：`python dca_btc.py`，默认基础金额 30 USDT，终端输出今日决策，并记录一次运行日志。【F:dca_btc.py†L1414-L1443】
- 环境变量：
  - `DATA_SOURCE`（`binance`/`okx`）控制现货 K 线主源；`DERIV_SOURCE`（`hyperliquid`/`binance`）控制衍生品指标主源。
  - `SIMULATE_BINANCE_FAIL=1` 可在本地模拟 Binance 失效以测试回退逻辑。【F:dca_btc.py†L130-L169】【F:dca_btc.py†L242-L268】
- 运行结果结构包含快照、估值/风险得分、倍数解释、数据源记录，可直接用于通知或回测。【F:dca_btc.py†L1220-L1247】

## notify_today.py 会做什么
- 调用 `run_today` 生成当日决策，构建推送文本并打印到控制台，然后写入运行日志（见下文）。【F:notify_today.py†L31-L72】【F:notify_today.py†L92-L111】
- 推送渠道：Telegram（需 `TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`）、Server 酱（`FT_SENDKEY`）、自定义 Webhook（`WEBHOOK_URLS`，逗号分隔）。任何推送失败都会被记录但不阻塞流程。【F:notify_today.py†L9-L48】【F:notify_today.py†L73-L111】
- 日志文件：`logs/dca_runs.csv`，每天覆盖同一日期的记录，包含价格、估值、风险、数据源、倍数、加仓/刹车标记等字段，写入异常会被吞掉以保证主流程安全。【F:dca_btc.py†L1254-L1374】

## 数据源选择与回退
- K 线：默认 Binance，失败自动回退 OKX；也可通过 `DATA_SOURCE` 设为 OKX 主、Binance 备。【F:dca_btc.py†L242-L268】
- 衍生品指标（Mark Price、Funding、OI）：默认 Hyperliquid，失败回退 Binance；可通过 `DERIV_SOURCE` 切换优先顺序。【F:dca_btc.py†L378-L439】【F:dca_btc.py†L442-L466】
- SSR：固定从 CoinGecko 读取 BTC 市值与稳定币篮子市值计算比值，并记录数据源标识。【F:dca_btc.py†L688-L720】
- OI 历史缓存：每次运行都会把 Hyperliquid 的 OI 名义价值写入 `hl_oi_history.json` 作为 Binance 历史缺失时的备份；文件位于仓库根目录，与 `dca_btc.py` 同级，Workflow 会在变更后自动提交。【F:dca_btc.py†L492-L520】

## 通过 backtest 优化参数
- 使用 `python backtest_dca.py --log logs/dca_runs.csv` 读取历史运行日志，套用新参数重新计算倍数、投入金额并给出总投入/平均倍数对比。【F:backtest_dca.py†L1-L81】
- 可调参数包含 `alpha`、`max_brake`、`z1/z2`、`edge1/edge2`、`trend_brake_scale_down` 等，命令行支持直接覆盖。【F:backtest_dca.py†L82-L115】
- 输出会显示原始与新配置下的投入总额、平均倍数以及倍数偏差超过 1 的天数，用于快速迭代策略配置。【F:backtest_dca.py†L44-L79】

## 其他重要说明
- 运行日志默认保存在 `logs/dca_runs.csv`，缺失目录会自动创建；新增记录时会补齐缺失字段并按日期去重更新。【F:dca_btc.py†L1254-L1375】
- 需要依赖 `requirements.txt` 中的第三方库，建议创建虚拟环境后 `pip install -r requirements.txt`。
- 如需每日自动推送，可在 CI/定时任务中执行 `python notify_today.py`，同时确保上述环境变量配置完整。

