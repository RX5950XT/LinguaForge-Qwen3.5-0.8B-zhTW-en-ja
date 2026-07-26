# LinguaForge — Qwen3.5-0.8B 繁中／英／日翻譯特化模型

把 `Qwen/Qwen3.5-0.8B`（873M）以 LoRA SFT 微調成 **繁體中文（臺灣）↔ 英文 ↔ 日文** 六方向翻譯特化小模型。全程在單張 RTX 3070 Ti 8GB 上完成。

## 動機

0.8B 這種尺寸的模型語意理解其實不差（COMET 82–86），但輸出的**字形與用語**不對：翻成中文時大量摻雜簡體字（ja→zhtw 高達 47%），日中互譯的表面品質也明顯偏弱。這類問題正是 SFT 最能補的部分。

## 環境需求

- Windows 11 / Python 3.12 / [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU ≥ 8GB VRAM（CUDA 12.8）

```powershell
uv sync                          # 主環境（訓練、推論、評測）
uv sync --project tools/comet    # COMET 評分隔離環境
```

> `unbabel-comet` 會鎖住舊版 transformers，與 Qwen3.5 所需的 5.x 衝突，因此獨立成子專案。

## 快速開始

```powershell
uv run python scripts/download_data.py        # 下載平行語料（冪等，可中斷重跑）
uv run python scripts/prepare_data.py         # 清洗 → data/sft/{train,dev}.jsonl
uv run python scripts/train_sft.py            # LoRA SFT
uv run python scripts/evaluate.py --tag sft-v1 --adapter outputs/sft
uv run --project tools/comet python tools/comet/score.py --tag sft-v1
```

## 專案結構

```
├─ CLAUDE.md / AGENTS.md      給 AI agent 的專案工作規範
├─ CONTEXT.md                 開發交接紀錄（接手先讀）
├─ README.md                  本文件
├─ docs/REPORT.md             研究報告（實驗結果全紀錄）
├─ release/       (git-ignored)   HF 上傳暫存：lora-v3/ merged-bf16/ gguf/ assets/ + MODEL_CARD.md
├─ pyproject.toml / uv.lock   主環境依賴（訓練 + 推論 + 評測）
│
├─ configs/                   訓練設定
│    ├─ sft_lora.yaml           0.8B r32/α64（v1、v2）
│    ├─ sft_lora_v3.yaml        0.8B r64/α128 + NEFTune（v3）
│    └─ sft_qlora_2b.yaml       2B NF4 4-bit QLoRA
│
├─ scripts/                   資料管線 + 訓練 + 評測
│    ├─ download_data.py        下載平行語料（冪等）
│    ├─ prepare_data.py         清洗 → s2twp → 污染閘 → sft jsonl
│    ├─ dump_eval_lines.py      抽 5 基準全量行建污染閘
│    ├─ train_sft.py            LoRA / QLoRA 訓練（含 NF4 分支）
│    ├─ bench_step.py           量 VRAM 防 8GB 靜默 fallback
│    ├─ evaluate.py             六方向翻譯 + chrF++/BLEU/洩漏（--nf4 / --int8 可選）
│    ├─ scoreboard.py           方向 × 基準矩陣
│    ├─ regression_guard.py     守 v2 底線
│    ├─ export_model.py         合併 LoRA → bf16 全模型 + 驗證
│    ├─ export_gguf.py          合併模型 → GGUF（--no-mtp）→ Q8_0/Q4_K_M（+ 可選 MTP draft）
│    └─ plot_loss.py            train/eval loss 雙曲線（--with matplotlib）
│
├─ tools/comet/               COMET 隔離子專案（獨立 uv 環境，鎖 transformers<4.58）
│    ├─ score.py                reference-based COMET
│    └─ kiwi.py                 CometKiwi reference-free（gated）
│
├─ data/          (git-ignored)
│    ├─ raw/                    原始平行語料 tsv
│    ├─ sft/                    train.jsonl + dev.jsonl（實際訓練吃這個）
│    ├─ flores200/             評測集快取
│    └─ eval_lines.txt          污染閘比對表
│
├─ outputs/       (git-ignored)   訓練產物
│    ├─ sft/ sft-v2/ sft-v3/ sft-2b-qlora/   各版 LoRA adapter（根 adapter + 最終 checkpoint）
│    └─ merged/                 v2 合併後全權重（推論 / 轉 GGUF 用）
│
├─ results/       (git-ignored)   評測結果，依模型版本分類
│    ├─ baseline/               官方零樣本地板（0.8B / 2B / +s2twp / NF4）
│    ├─ v1/ v2/ v3/             0.8B 微調三代（<tag>-<benchmark>.json）
│    ├─ v3-2b/                  2B QLoRA 微調
│    ├─ hyp/                    各方向 src/ref/hyp 譯文（COMET 讀這個）
│    ├─ scoreboard.md           六模型 × 五基準總表
│    └─ data_stats.json         資料清洗統計
│
├─ logs/          執行 log 依類型分：data/ bench/ train/ eval/ comet/ export/
└─ tasks/         todo.md（進度）、lessons.md（踩坑教訓）
```

> `results/*.json` 檔名沿用 `<tag>-<benchmark>.json` 慣例，分類子目錄由 `scoreboard.py` / `regression_guard.py` 遞迴讀取（`rglob`）。

## 資料

六個方向各 75,000 句，合計 45 萬筆訓練樣本（dev 每方向 200 句）。

| 來源 | 用於 | 說明 |
|---|---|---|
| COCT（Taiwan Panorama） | en↔zhtw | 原生臺灣正體，品質最佳 |
| TED2020 | en↔zhtw、ja↔zhtw、en↔ja | 演講字幕，乾淨完整句 |
| WikiMatrix / JParaCrawl-filtered / Tatoeba / News-Commentary | en↔ja | 書面體乾淨語料，貼 FLORES domain |
| OPUS-100 | en↔zhtw | 大量但雜訊較多 |
| OpenSubtitles / WikiMatrix / News-Commentary | ja↔zhtw | 補稀缺方向 |

> en↔ja 初版用 OPUS-100，因其口語/雜訊與 FLORES 書面體不匹配拖累日文方向；改用上列乾淨書面語料重訓後全面提升（見下方結果）。

清洗流程：控制字元正規化 → 字幕雜訊剝除（講者標記、歌詞符號、對齊錯誤的重複句）→ 長度與長度比過濾 → 語言驗證（假名／漢字／ASCII 比例）→ **OpenCC `s2twp` 統一臺灣正體** → 去重。統計寫入 `results/data_stats.json`。

## 評測

FLORES-200 devtest，六方向各 500 句，三項指標：

- **chrF++ / BLEU**（sacrebleu，日文用 `ja-mecab`、中文用 `zh` tokenizer）
- **COMET**（`Unbabel/wmt22-comet-da`）
- **簡體洩漏率**（OpenCC `s2t` round-trip 檢測）

### 基線（微調前）

| 方向 | chrF++ | BLEU | COMET | 簡體洩漏 |
|---|---|---|---|---|
| en→zhtw | 20.9 | 24.8 | 86.0 | 20.2% |
| zhtw→en | 48.0 | 19.4 | 84.4 | — |
| en→ja | 18.9 | 16.9 | 82.9 | — |
| ja→en | 45.2 | 15.9 | 83.8 | — |
| ja→zhtw | 11.6 | 10.3 | 82.6 | 47.0% |
| zhtw→ja | 14.9 | 11.8 | 83.2 | — |

### 0.8B 微調後（sft-v2，六方向 FLORES）

| 方向 | chrF++ | BLEU | COMET | 簡體洩漏 |
|---|---|---|---|---|
| en→zhtw | 18.9 | 24.9 | 84.6 | **6.4%** |
| zhtw→en | 47.9 | 21.0 | 84.3 | — |
| en→ja | 21.6 | 19.8 | **85.8** | — |
| ja→en | 48.0 | 20.1 | 85.1 | — |
| ja→zhtw | 15.4 | 20.8 | 84.6 | **6.4%** |
| zhtw→ja | 16.6 | 14.4 | 83.1 | — |

核心目標達成——簡體洩漏暴跌（en→zhtw 20.2%→6.4%、ja→zhtw 47.0%→6.4%），
COMET 語意均分 **83.8→84.6 反超基線**（六方向 4 升 2 持平），日文方向 en→ja +2.9。

### v3 多領域強化與 2B 對照（最終定案）

v3 擴充到 5 個基準（FLORES／NTREX／WMT22／ALT／TICO-19，涵蓋維基／新聞／字幕／醫療），
資料多元化到 77 萬句並加污染閘，並行訓練 0.8B（r64+NEFTune）與 2B QLoRA。**完整分數表見
[`results/scoreboard.md`](results/scoreboard.md)**。三個結論：

1. **0.8B 微調有正價值**：v3 簡體洩漏再降（FLORES en→zhtw 5.2%、ja→zhtw 6.0%），
   最弱的 zhtw→ja 跨三領域一致上升，六方向回歸守門全過。
2. **2B 的最佳解不是微調，而是官方 + 後處理**：補測官方 Qwen3.5-2B 零樣本後發現它的 COMET
   每個基準都高於我們的 QLoRA 微調版，只是簡體洩漏 16～51%。套上既有的 OpenCC `s2twp`
   輸出後處理即可壓到 ≤5.4%（過閘）→ **「官方 Qwen3.5-2B + s2twp」是品質最高、零訓練的出貨組合**
   （FLORES en→zhtw COMET 88.21）。
3. **8GB 的量化稅**：2B 微調版看似落後，NF4 2×2 對照顯示同精度下微調≈中性，落差大半來自
   被 8GB 顯存逼用的 4-bit QLoRA 量化損失，非微調本身。真正判定需雲端全精度 bf16 重訓 2B。

> COMET 對簡繁不敏感（會獎勵洩漏簡體卻流暢的輸出），故 COMET 與簡體洩漏率必須並看。

## 訓練設定

LoRA target 涵蓋標準注意力與 Qwen3.5 的線性注意力層（`in_proj_qkv/z/a/b`、`out_proj`）；bf16、packing、lr 1e-4 cosine。三份 config：

| config | 模型 | 設定 | adapter |
|---|---|---|---|
| `sft_lora.yaml` | 0.8B | r32/α64、2 epoch（v1/v2） | `outputs/sft`、`outputs/sft-v2` |
| `sft_lora_v3.yaml` | 0.8B | r64/α128 + NEFTune=5、1 epoch | `outputs/sft-v3` |
| `sft_qlora_2b.yaml` | 2B | NF4 4-bit QLoRA、bs1×768+ga32 | `outputs/sft-2b-qlora` |

> **8GB 顯存陷阱**：Qwen3.5 的 vocab 高達 248K，logits 物化使 VRAM 隨 batch×seq 暴增；2B 只能靠 NF4 塞進 8GB。Windows 超顯存不會 OOM，NVIDIA 驅動會靜默 fallback 到系統記憶體、速度掉到 1/5 且無警告——訓練前務必用 `scripts/bench_step.py` 量 VRAM <8GB。
> 另 `flash-linear-attention` 無 Windows wheel，線性注意力走 torch fallback（較慢但可用）。

## 發布

開源 **0.8B v3**，大檔上 Hugging Face、程式碼留 GitHub（本倉庫）：

| 位置 | 內容 |
|---|---|
| **Hugging Face** | LoRA adapter、合併 bf16 全模型、GGUF（Q8_0 / Q4_K_M / f16 / MTP draft）、`MODEL_CARD.md`、loss 曲線 |
| **GitHub（本倉庫）** | 核心程式碼、config、`docs/REPORT.md`；語料不 re-host，用 `download_data.py` + `prepare_data.py` 重建 |

打包流程（產物皆進 git-ignored 的 `release/`）：

```powershell
uv run python scripts/export_model.py --adapter outputs/sft-v3 --out release/merged-bf16   # 合併 bf16
uv run python scripts/export_gguf.py --llama-cpp <llama.cpp> --quantize-bin <llama-quantize> --mtp
uv run --with matplotlib python scripts/plot_loss.py                                         # loss 曲線
```

GGUF 實測（RTX 3070 Ti，Q8_0）：GPU `-ngl 99` ~128–190 t/s、CPU ~29 t/s，皆輸出正確台灣正體。
詳見 `release/lora-v3/MODEL_CARD.md`（含推論範例與 `eos_token_id=[248046,248044]`、`enable_thinking:false` 等關鍵旗標）。

## 授權

模型與程式碼依 Apache-2.0；各語料授權依其原始來源（本倉庫僅發布微調權重，不轉發語料）。
