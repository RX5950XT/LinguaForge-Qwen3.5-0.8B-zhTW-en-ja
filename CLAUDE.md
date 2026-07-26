# LinguaForge-Qwen3.5-0.8B-zhTW-en-ja

Qwen3.5-0.8B 翻譯特化微調（zh-TW ↔ en ↔ ja 六方向，LoRA SFT，本機 RTX 3070 Ti 8GB）。

## 環境

- Windows 11 + Python 3.12 + uv；主環境 `uv sync`，COMET 隔離環境 `uv sync --project tools/comet`
- 模型 `Qwen/Qwen3.5-0.8B`：multimodal 架構，用 `AutoModelForImageTextToText` 載入，transformers ≥5.x
- `unbabel-comet` 會鎖舊版 transformers，**絕不可**裝進主環境

## 常用指令

```powershell
uv run python scripts/download_data.py                 # 下載語料（冪等）
uv run python scripts/prepare_data.py                  # 清洗 → data/sft/*.jsonl
uv run python scripts/test_prepare_data.py             # 清洗函數自檢
uv run python scripts/evaluate.py --tag <tag> [--adapter <dir>] [--full]
uv run --project tools/comet python tools/comet/score.py --tag <tag>
uv run python scripts/train_sft.py [--max-steps N]
```

## 關鍵知識

- LoRA target 含線性注意力層：`in_proj_qkv/z/a/b`、`out_proj`（Qwen3.5 混合 linear attention）
- 訓練資料 chat messages 格式；train_sft.py 轉 prompt/completion 讓 TRL 遮罩 prompt loss
- 評測集：FLORES-200 devtest（Meta 公開 tarball，自動快取 data/flores200；FLORES+ gated 拿不到）
- 指標：chrF++ / BLEU（sacrebleu）+ COMET（wmt22-comet-da）+ 簡體洩漏率（OpenCC s2t round-trip）
- flash-linear-attention 未裝（Windows），linear attention 走 torch fallback，速度較慢但可用
- **推論務必設 `eos_token_id=[248046, 248044]`**（im_end + endoftext）：SFT 版學會用 im_end 收尾，
  但 config 預設 eos 是 endoftext，不設會失控重複。見 evaluate.py `stop_token_ids`
