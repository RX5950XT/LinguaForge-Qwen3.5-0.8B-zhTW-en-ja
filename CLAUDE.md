# LinguaForge-Qwen3.5-0.8B-zhTW-en-ja

Qwen3.5-0.8B 翻譯特化微調（zh-TW ↔ en ↔ ja 六方向，LoRA SFT，本機 RTX 5060 Ti 16GB）。

## 環境

- Windows 11 + Python 3.12 + uv；主環境 `uv sync`，COMET 隔離環境 `uv sync --project tools/comet`
- 模型 `Qwen/Qwen3.5-0.8B`：multimodal 架構，用 `AutoModelForImageTextToText` 載入，transformers ≥5.x
- `unbabel-comet` 會鎖舊版 transformers，**絕不可**裝進主環境

## 常用指令

```powershell
uv run python scripts/download_data.py                 # 下載語料（冪等）
uv run python scripts/prepare_data.py --limit 20000    # 清洗 → data/sft/*.jsonl（--limit 必給！）
uv run python scripts/test_prepare_data.py             # 清洗函數自檢
uv run python scripts/evaluate.py --tag <tag> [--adapter <dir>] [--full]
uv run --project tools/comet python tools/comet/score.py --tag <tag>-flores   # 全名，不是版本號
uv run python scripts/eval_capability.py --tag <tag> [--adapter <dir>] --axis all
uv run python scripts/eval_bench.py --tag <tag> [--adapter <dir>]   # BELEBELE + 知識
uv run python scripts/train_sft.py [--config configs/sft_lora_v5f.yaml] [--max-steps N]
```

## 出貨標準

**硬閘**（任一 FAIL 不得出貨，`regression_guard.py` 已覆蓋前兩項）：

| 項目 | 門檻 |
|---|---|
| 簡體洩漏 en→zhtw / ja→zhtw | ≤ baseline + 0.3 |
| 六方向 COMET | 每個 ≥ base |
| 翻譯機率（`eval_capability --axis general`） | ≤ 5.0% |
| BELEBELE / 知識 三語（`eval_bench.py`） | 各 ≥ base − 3.0 |
| `--axis doc` 尾段譯出比 `tail_ratio_median` | ≥ 0.80（六向皆須） |
| `--axis doc` 腰斬率 `truncated_pct` | ≤ 5%（六向皆須） |

**目標**（達成即收工）：COMET AVG ≥ 87.00、通用能力 n=90 ≥ base − 3.0。

**停止規則**：連續兩版 COMET AVG 提升 < 0.10 → 停，出貨當前最佳版本。
無限加碼跑下去不是嚴謹，是沒有驗收條件。

doc 軸的閘用**絕對值不跟 base 比**：base 自己尾段會超譯（tail_ratio 1.10~1.65、
行數比 1.4~1.5），拿它當地板等於在問「候選有沒有跟 base 一樣多話」。
`completeness_median`（整篇字元比）已降為診斷欄位——它把「行文精簡／多吐垃圾行／
尾段腰斬」混成一個數字，三者互相抵消。判腰斬只看 `tail_ratio_median` 與 `truncated_pct`。

## 關鍵知識

- LoRA target 含線性注意力層：`in_proj_qkv/z/a/b`、`out_proj`（Qwen3.5 混合 linear attention）
- 訓練資料 chat messages 格式；train_sft.py 轉 prompt/completion 讓 TRL 遮罩 prompt loss
- 評測集：FLORES-200 devtest（Meta 公開 tarball，自動快取 data/flores200；FLORES+ gated 拿不到）
- 指標：chrF++ / BLEU（sacrebleu）+ COMET（wmt22-comet-da）+ 簡體洩漏率（OpenCC s2t round-trip）
- flash-linear-attention 0.5.1 + triton-windows **已裝且生效**（gated delta rule 走 Triton kernel）。
  transformers 那句 "fast path is not available" 只是在抱怨 `causal_conv1d` 沒裝——modeling_qwen3_5.py
  是逐個 op 各自 fallback，缺 causal_conv1d 只讓 depthwise conv 退回 `nn.Conv1d`（cuDNN，本來就快）
  與解碼時每 token 多 4 個 kernel launch。PyPI 上 causal-conv1d **只有 sdist 沒有任何 wheel**，
  要編就得補 CUDA 12.8 toolkit（本機是 13.3，跟 torch cu128 不同大版本，`_check_cuda_version` 會擋）
- **推論務必設 `eos_token_id=[248046, 248044]`**（im_end + endoftext）：SFT 版學會用 im_end 收尾，
  但 config 預設 eos 是 endoftext，不設會失控重複。見 evaluate.py `stop_token_ids`
- **只要最後一格 logits 就一定要 `logits_to_keep=1`**：vocab 248K，不設會實體化
  `[B, L, 248064]` 整張（B=16、L=800 → 6.3GB，實測 15.85/16.31 GB 貼著 OOM）。
  訓練沒踩到是因為 `token_budget` 早就壓死每個 micro-batch 的 token 數
- GGUF 轉換用 `D:/Workspace/AI_training/llama.cpp-latest`（本機另一份 b8189 太舊，
  不認 Qwen3.5 的 BPE pre-tokenizer）。llama-cli 執行時**必須** `--reasoning off
  --reasoning-budget 0`——舊的 `--chat-template-kwargs enable_thinking` 已被靜默忽略，
  thinking 會以雜訊前綴洩進譯文。CJK 提示詞用 `-f prompt.txt`，`-p` 會被 cp950 打壞
- COMET 的 base 對照只認 `results/baseline/base-full-flores.json`（n=1012 + 現行 `DECODE`）。
  宣告某方向勝負前先跑 `tools/comet/paired_bootstrap.py`，CI 跨 0 就是打平
- 選擇題基準（`eval_bench.py`）必須輪轉選項去偏：0.8B 對答案位置有強先驗
  （實測 base 押同一字母 43%、v5f 63%，隨機應為 ~27%），單輪 acc 量到的是先驗不是知識。
  比選項文字 logprob 那條路實測貼著隨機基準，已排除
