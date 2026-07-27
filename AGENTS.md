# AGENTS

本專案通用規範同 `CLAUDE.md`（環境、指令、關鍵知識），交接進度見 `CONTEXT.md`。

要點速查：

- 套件管理一律 `uv`；主環境與 `tools/comet` 隔離環境不可混裝
- 模型載入用 `AutoModelForImageTextToText`（transformers ≥5.x）
- 訓練/評測腳本都在 `scripts/`，設定在 `configs/sft_lora.yaml`
- 評測產出：`evaluate.py` → `results/<tag>-<benchmark>.json`（跑完歸檔進 `results/<版本>/`，
  `scoreboard.py` 用 rglob 所以子目錄照樣讀得到）；`eval_capability.py` → `results/capability/<tag>.json`。
  對照基準在 `results/baseline/`
- flash-linear-attention + triton-windows **已裝且生效**，gated delta rule 走 Triton kernel。
  transformers 開場那句 "fast path is not available" 只是在講 `causal_conv1d` 沒裝
  （逐 op fallback，影響僅止於 depthwise conv 與解碼時多幾個 kernel launch），**不必理會**，
  也不要據此以為 linear attention 在跑 torch fallback
- 發布打包在 `release/`（gitignored，上 HF）；GGUF 轉換須 `--no-mtp`（否則 runtime 缺 `blk.24` tensor）
