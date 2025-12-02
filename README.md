# BTC DCA Bot

## Hyperliquid OI 缓存文件放在哪里？

- `hl_oi_history.json` 与 `dca_btc.py` 位于同一目录（仓库根目录）。
- 脚本通过 `OI_CACHE_FILE = os.path.join(os.path.dirname(__file__), "hl_oi_history.json")` 读取/写入，因此无论工作目录在哪里，都会定位到这里。
- GitHub Actions 默认在 `${{ github.workspace }}` 运行，我们在 workflow 里也显式设置了 `working-directory`，所以线上/本地都用同一份文件。
- Workflow 每天运行后会检测这个文件是否变化，并自动提交回仓库，确保缓存可持久化。
