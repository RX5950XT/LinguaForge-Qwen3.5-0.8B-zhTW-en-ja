# LinguaForge — Qwen3.5-0.8B 繁中／英／日翻譯特化模型

把 `Qwen/Qwen3.5-0.8B`（873M）以 LoRA SFT 微調成 **繁體中文（臺灣）↔ 英文 ↔ 日文** 六方向翻譯特化小模型。全程在單張 RTX 3070 Ti 8GB 上完成。

## 動機

0.8B 這種尺寸的模型語意理解其實不差（COMET 82–86），但輸出的**字形與用語**不對：翻成中文時大量摻雜簡體字（FLORES ja→zhtw 高達 **45.6%**），日中互譯的表面品質也明顯偏弱。這類問題正是 SFT 最能補的部分。

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
uv run python scripts/prepare_data.py         # 清洗 + s2twp + 污染閘 → data/sft/{train,dev}.jsonl
uv run python scripts/train_sft.py --config configs/sft_lora_v3.yaml     # LoRA SFT（~8h on 3070 Ti）
uv run python scripts/evaluate.py --tag v3 --benchmark all --adapter outputs/sft-v3
uv run --project tools/comet python tools/comet/score.py --tag v3-flores
uv run python scripts/scoreboard.py --tags v3                            # 方向 × 基準總表
```

## 專案結構

大檔（語料、權重、GGUF）不進版控——見下方[「哪些東西不在這個倉庫」](#哪些東西不在這個倉庫)。

```
├─ README.md                  本文件
├─ CLAUDE.md / AGENTS.md      給 AI agent 的專案工作規範
├─ CONTEXT.md                 開發交接紀錄（接手先讀）
├─ LICENSE                    Apache-2.0
├─ pyproject.toml / uv.lock   主環境依賴（訓練 + 推論 + 評測）
│
├─ docs/
│    ├─ REPORT.md               研究報告：實驗全紀錄、多領域診斷、量化稅、訓練成本
│    └─ assets/loss_curve.png   v3 SFT train/eval loss 雙曲線
│
├─ configs/                   訓練設定
│    ├─ sft_lora.yaml           0.8B r32/α64、2 epoch（v1、v2）
│    ├─ sft_lora_v3.yaml        0.8B r64/α128 + NEFTune、1 epoch（v3 主線）
│    └─ sft_qlora_2b.yaml       2B NF4 4-bit QLoRA（bs1×768，能力 vs 資料診斷）
│
├─ scripts/                   資料管線 → 訓練 → 評測 → 打包，一條線
│    ├─ download_data.py        下載 21 份平行語料（冪等，可中斷重跑）
│    ├─ prepare_data.py         清洗 → s2twp 台灣正體 → 污染閘 → data/sft/*.jsonl
│    ├─ test_prepare_data.py    清洗函數自檢（無框架，直接跑）
│    ├─ dump_eval_lines.py      抽 5 基準全量行 → 污染閘比對表
│    ├─ inspect_sft.py          抽看訓練樣本長相
│    ├─ count_tokens.py         估 token 預算
│    ├─ bench_step.py           跑幾步量 VRAM/吞吐，選 batch×seq（防 8GB 靜默 fallback）
│    ├─ train_sft.py            LoRA / QLoRA 訓練（含 NF4 分支）
│    ├─ evaluate.py             六方向翻譯 + chrF++/BLEU/洩漏率（--nf4 / --int8 可選）
│    ├─ scoreboard.py           方向 × 基準矩陣（抓「FLORES 好看但新聞崩」）
│    ├─ regression_guard.py     斷言不破 v2 底線，過了才算候選
│    ├─ smoke_test.py           改完先冒煙
│    ├─ export_model.py         合併 LoRA → bf16 全模型 + 抽測驗證
│    ├─ verify_merged.py        合併模型六方向抽測
│    ├─ export_gguf.py          bf16 → GGUF（主檔須 --no-mtp）→ Q8_0/Q4_K_M（+ MTP draft）
│    └─ plot_loss.py            loss 雙曲線（uv run --with matplotlib，不污染主環境）
│
├─ tools/comet/               COMET 隔離子專案（獨立 uv 環境，鎖 transformers<4.58）
│    ├─ score.py                reference-based COMET，分數寫回 results/*.json
│    └─ kiwi.py                 CometKiwi reference-free（gated、非商用，僅評測）
│
├─ results/                   評測結果（進版控的實驗證據）
│    ├─ scoreboard.md           六模型 × 五基準總表
│    ├─ data_stats.json         每來源清洗統計 + 方向/領域實際配比
│    ├─ baseline/               官方零樣本地板（0.8B / 2B / +s2twp / NF4 / int8）
│    ├─ v1/ v2/ v3/             0.8B 微調三代（<tag>-<benchmark>.json）
│    ├─ v3-2b/                  2B QLoRA 微調
│    ├─ v3/trainer_state.json   v3 loss 歷史（plot_loss.py 讀這個；outputs/ 刪了也還在）
│    └─ hyp/      (git-ignored) 各方向 src/ref/hyp 譯文，35MB，evaluate.py 可重跑產生
│
├─ tasks/         todo.md（進度勾選）、lessons.md（踩坑教訓）
├─ data/          (git-ignored) raw/ 原始 tsv、sft/ 訓練 jsonl、flores200/ 評測快取
├─ outputs/       (git-ignored) 訓練產物：sft/ sft-v2/ sft-v3/ sft-2b-qlora/ merged/
├─ release/       (git-ignored) HF 上傳暫存，見下
└─ logs/          (git-ignored) 執行 log：data/ bench/ train/ eval/ comet/ export/
```

> `results/*.json` 沿用 `<tag>-<benchmark>.json` 慣例；分類子目錄由 `scoreboard.py` / `regression_guard.py` 遞迴讀取（`rglob`），放哪層都找得到。

### 哪些東西不在這個倉庫

| 排除 | 大小 | 怎麼拿回來 |
|---|---|---|
| `data/` | 1.9 GB | `download_data.py` + `prepare_data.py` 重建（語料授權各異，不 re-host） |
| `outputs/` | 3.4 GB | 重訓；**v3 的 adapter 與 loss 歷史已分別上 HF 與 `results/v3/trainer_state.json`，不會遺失** |
| `release/` | 5.1 GB | Hugging Face（見[發布](#發布)） |
| `results/hyp/` | 35 MB | `evaluate.py --tag <tag>` 重跑 |
| `logs/` | 3 MB | 純執行紀錄；其中有價值的數字（loss、peak VRAM、wall-clock、8GB 選型）已全數提煉進 `results/` 與 `docs/REPORT.md` |

## 資料

v3 六方向各 ~130k、合計 **773,389** 句（dev 每方向 200）。每來源設佔比上限並標領域，避免單一語料壟斷某方向。

| 來源 | 用於 | 說明 |
|---|---|---|
| COCT（Taiwan Panorama） | en↔zhtw | 原生臺灣正體，品質最佳 |
| TED2020 | en↔zhtw、ja↔zhtw、en↔ja | 演講字幕，乾淨完整句 |
| WikiMatrix / JParaCrawl / KFTT / Tatoeba / News-Commentary | en↔ja | 書面體乾淨語料 |
| GlobalVoices（原生繁新聞）/ KDE4（原生繁 IT） | en↔zhtw、ja↔zhtw | v3 新增，補新聞與在地化語域 |
| OpenSubtitles / MTNT | 僅 →en 的源側 | v3 新增，救 into-English；只進源側以護洩漏戰果 |
| OPUS-100 | en↔zhtw | 量大但雜訊較多 |

> en↔ja 初版用 OPUS-100，其口語/雜訊拖累日文方向；改用上列乾淨書面語料重訓後全面提升。

清洗流程：控制字元正規化 → 字幕雜訊剝除（講者標記、歌詞符號、對齊錯誤的重複句）→ 長度與長度比過濾 → 語言驗證（假名／漢字／ASCII 比例）→ **OpenCC `s2twp` 統一臺灣正體** → 去重 → **免污染閘**（hash 5 個評測基準的個別語言行共 33,984 行，訓練對任一側命中即丟）。每來源統計與方向/領域實際配比寫入 `results/data_stats.json`。

## 評測

FLORES-200 devtest，六方向各 500 句，三項指標：

- **chrF++ / BLEU**（sacrebleu，日文用 `ja-mecab`、中文用 `zh` tokenizer）
- **COMET**（`Unbabel/wmt22-comet-da`）
- **簡體洩漏率**（OpenCC `s2t` round-trip 檢測）

### 發布版 v3（0.8B）vs 官方零樣本

| 方向 | chrF++ (base→v3) | BLEU (base→v3) | COMET (base→v3) | 簡體洩漏 (base→v3) |
|---|---|---|---|---|
| en→zhtw | 20.92 → 19.28 | 24.98 → 25.91 | 85.92 → 85.08 | **20.8% → 5.2%** |
| zhtw→en | 47.94 → 48.82 | 19.33 → 21.09 | 84.40 → 84.47 | — |
| en→ja | 18.79 → 20.96 | 17.43 → 19.31 | 83.20 → **85.55** | — |
| ja→en | 45.37 → 48.06 | 16.01 → 19.80 | 83.82 → 84.89 | — |
| ja→zhtw | 11.66 → 15.32 | 10.42 → 20.83 | 82.69 → **84.74** | **45.6% → 6.0%** |
| zhtw→ja | 14.95 → 17.12 | 12.05 → 15.41 | 83.07 → 84.09 | — |

核心目標達成——簡體洩漏暴跌，COMET 語意均分 **83.85 → 84.80 反超基線**（六方向 5 升 1 微降），
en→ja +2.35。唯一 tradeoff 是 en→zhtw：輸出台灣用語與 FLORES 參考的特定字詞有落差，
表面分略降，換來洩漏 20.8%→5.2%。

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

實際成本（單張 RTX 3070 Ti 8GB）：v3-0.8B **peak VRAM 4.01GB、8.0 小時**（2390 步 / 57.9M tokens，final eval_loss 1.988）；2B NF4 QLoRA 5.10GB / 17.3 小時。

![v3 SFT loss](docs/assets/loss_curve.png)

> **8GB 顯存陷阱**：Qwen3.5 的 vocab 高達 248K，logits 物化使 VRAM 隨 batch×seq 暴增；2B 只能靠 NF4 塞進 8GB。Windows 超顯存不會 OOM，NVIDIA 驅動會靜默 fallback 到系統記憶體、速度掉到 1/5 且無警告——訓練前務必用 `scripts/bench_step.py` 量 VRAM <8GB。
> 另 `flash-linear-attention` 無 Windows wheel，線性注意力走 torch fallback（較慢但可用）。

## 發布

開源 **0.8B v3**，大檔上 Hugging Face、程式碼留 GitHub（本倉庫）：

| 位置 | 內容 |
|---|---|
| **Hugging Face** | LoRA adapter、合併 bf16 全模型、GGUF（Q8_0 / Q4_K_M / f16 / MTP draft）、模型卡、loss 曲線 |
| **GitHub（本倉庫）** | 核心程式碼、config、評測結果、`docs/REPORT.md`；語料不 re-host，用 `download_data.py` + `prepare_data.py` 重建 |

`release/` 就是 HF model repo 的完整鏡像，目錄結構即上傳後的樣子——adapter 放根目錄（`library_name: peft` 要求），模型卡是 **`README.md`**（HF 只渲染這個檔名，且需 YAML frontmatter）：

```
release/
├─ README.md                     模型卡（frontmatter: apache-2.0 / base_model / pipeline_tag: translation）
├─ .gitattributes                *.safetensors、*.gguf 走 LFS
├─ adapter_model.safetensors     LoRA r64/α128（173MB）
├─ adapter_config.json           base = Qwen/Qwen3.5-0.8B
├─ tokenizer.json / tokenizer_config.json / chat_template.jinja
├─ assets/loss_curve.png         模型卡內嵌
├─ merged-bf16/                  合併全模型 1.7GB（免裝 peft，可直接轉檔）
└─ gguf/                         Q8_0 775M / Q4_K_M 505M / f16 1.5G / mtp-f16 496M
```

打包流程：

```powershell
uv run python scripts/export_model.py --adapter outputs/sft-v3 --out release/merged-bf16   # 合併 bf16
uv run python scripts/export_gguf.py --llama-cpp <llama.cpp> --quantize-bin <llama-quantize> --mtp
uv run --with matplotlib python scripts/plot_loss.py --out release/assets/loss_curve.png
hf upload <repo-id> release/ .                                                             # 上傳
```

GGUF 實測（RTX 3070 Ti，Q8_0）：GPU `-ngl 99` ~128–190 t/s、CPU ~29 t/s，皆輸出正確台灣正體。
推論範例與 `eos_token_id=[248046,248044]`、`enable_thinking:false` 等關鍵旗標詳見 `release/README.md`。

## 授權

模型與程式碼依 Apache-2.0；各語料授權依其原始來源（本倉庫僅發布微調權重，不轉發語料）。
