# AGENTS

本專案通用規範同 `CLAUDE.md`（環境、指令、**出貨標準**、關鍵知識），交接進度見 `CONTEXT.md`。

要點速查：

- 套件管理一律 `uv`；主環境與 `tools/comet` 隔離環境不可混裝
- 模型載入用 `AutoModelForImageTextToText`（transformers ≥5.x）
- 訓練/評測腳本都在 `scripts/`，設定在 `configs/sft_lora*.yaml`（每版一個檔，不改舊檔）
- **出貨要過 `CLAUDE.md` 的硬閘**，並遵守停止規則（連兩版 COMET AVG Δ < 0.10 就收工）。
  六條硬閘已全部機器化：`uv run python scripts/regression_guard.py --candidate <tag>`
  （exit 0=過／1=FAIL／**2=缺值，沒跑過的閘不算過**）。**不要人工核對表格判出貨。**
- 評測產出：`evaluate.py` → `results/<tag>-<benchmark>.json`（跑完歸檔進 `results/<版本>/`，
  讀取端一律 rglob 所以子目錄照樣讀得到）；`eval_capability.py` → `results/capability/<tag>.json`；
  `eval_bench.py` → `results/bench/<tag>.json`。對照基準在 `results/baseline/`，
  慣例見 `results/README.md`
- flash-linear-attention + triton-windows **已裝且生效**，gated delta rule 走 Triton kernel。
  transformers 開場那句 "fast path is not available" 只是在講 `causal_conv1d` 沒裝
  （逐 op fallback，影響僅止於 depthwise conv 與解碼時多幾個 kernel launch），**不必理會**，
  也不要據此以為 linear attention 在跑 torch fallback
- 發布打包在 `release/`（gitignored，上 HF）；GGUF 轉換須 `--no-mtp`（否則 runtime 缺 `blk.24` tensor）
