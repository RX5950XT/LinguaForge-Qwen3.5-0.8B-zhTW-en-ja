# LinguaForge — Qwen3.5-0.8B 繁中／英／日翻譯特化模型

LinguaForge 是以 `Qwen/Qwen3.5-0.8B`（約 873M 參數）為基底、經 LoRA 監督式微調（SFT）的**翻譯特化**小模型，支援 **繁體中文（臺灣）↔ 英文 ↔ 日文** 六個方向。目標輸出為**臺灣正體中文**（非簡體），並在單張消費級 GPU 上即可完成訓練與推論（發布版 v5e：RTX 5060 Ti 16GB、峰值約 4.0GB VRAM、訓練約 10.5 小時）。

| 資源 | 連結 |
|---|---|
| **模型權重（v5e）** | [Hugging Face · `RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja`](https://huggingface.co/RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja)（LoRA、合併 bf16、GGUF、[模型卡](https://huggingface.co/RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja/blob/main/README.md)） |
| **訓練／評測程式碼** | 本倉庫；本機打包鏡像見 `release/`（gitignored） |

## 動機

在 0.8B 規模下，官方基底模型的語意理解已有一定水準（零樣本 COMET 均分約 84.64），但中文輸出的**字形與用語**常不符臺灣使用習慣：FLORES 上 ja→zhtw 方向約有 **43.6%** 的句子混入簡體字。這類問題適合以 SFT 補強。發布版 v5e 將兩個 →zhtw 方向的簡體洩漏分別壓至 **1.09%** 與 **0.69%**，同時六方向語意分數整體上升。

## 評測結果（v5e）

基準：FLORES-200 devtest **全量 n=1012**；解碼為 beam search（`num_beams=4`）並依目標語言套用防重複參數。對照組為**相同樣本數、相同解碼設定**下的官方 `Qwen/Qwen3.5-0.8B`。

| 方向 | chrF++ (base → v5e) | BLEU (base → v5e) | COMET (base → v5e) | 簡體洩漏 (base → v5e) |
|---|---|---|---|---|
| en→zhtw | 19.38 → 20.26 | 25.57 → 28.90 | 86.32 → 86.23 | **10.18% → 1.09%** |
| zhtw→en | 47.78 → 50.33 | 19.34 → 23.72 | 84.79 → **85.27** | — |
| en→ja | 20.26 → 24.15 | 19.81 → 22.65 | 86.17 → **88.44** | — |
| ja→en | 45.67 → 50.17 | 16.67 → 22.70 | 84.85 → **86.20** | — |
| ja→zhtw | 10.53 → 16.60 | 10.38 → 23.07 | 81.44 → **86.10** | **43.58% → 0.69%** |
| zhtw→ja | 14.80 → 18.48 | 12.97 → 16.93 | 84.24 → **87.14** | — |
| **均值** | 26.40 → **30.00** | 17.46 → **22.60** | 84.64 → **86.56** | |

- COMET 均分約 **+1.92**；chrF++ 與 BLEU 六方向皆上升。
- **en→zhtw 語意與官方統計持平**：paired bootstrap 95% CI 約 [−0.414, +0.225]（跨 0）。官方該方向的 COMET 是在約 10% 簡體洩漏下取得的（COMET 對簡繁字形不敏感），兩邊產出的字形品質並不相同。
- 通用能力（外部公開基準）：BELEBELE 三語 **57.28 / 52.36 / 62.81**、知識軸（TMMLU+ / MMMLU-JA / MMLU）**30.53 / 36.64 / 43.81**，六格皆不低於官方基底。
- 長文件翻譯：行數對齊中位約 1.000、腰斬率約 0–4%；早期版本「長文只翻前兩段」的問題已消除。

完整實驗紀錄與方法論見 [`docs/REPORT.md`](docs/REPORT.md)。

## 推論參數與接入注意

正式評測與建議的 transformers 推論設定如下（實作參考 `scripts/evaluate.py`）：

| 參數 | 建議值 | 說明 |
|---|---|---|
| `num_beams` | **4** | greedy 品質較差，僅適快速試跑 |
| `length_penalty` | **1.2** | 網格搜尋後的建議值（較 1.0 略佳） |
| `do_sample` | **false** | 翻譯採確定性解碼 |
| `eos_token_id` | **`[248046, 248044]`** | `<\|im_end\|>` 與 `<\|endoftext\|>` 皆須視為結束；缺一會導致譯完後繼續生成 |
| 目標語 en / ja | `repetition_penalty=1.1` + `no_repeat_ngram_size=4` | 抑制句級重複（尤其長口語、社群文 → 英文） |
| 目標語 zhtw | **僅** `no_repeat_ngram_size=4` | **不應對繁中目標使用 `repetition_penalty`**，否則簡體洩漏可能明顯上升 |

**常見異常多半來自設定不符，而非模型「只有長文會壞」：**

| 現象 | 常見原因 | 處理 |
|---|---|---|
| 譯完後灌水至 `max_new_tokens` | 只設定單一 EOS | 使用雙 EOS |
| 長口語／社群文 → 英文同一句重複 | 未開 `no_repeat_ngram_size`；或 GGUF 僅 greedy | transformers 依上表；GGUF 可加 `--repeat-penalty` 並分段 |
| 譯文前綴出現思考句、選項字母等雜訊 | 未關閉 thinking | llama.cpp：`--reasoning off --reasoning-budget 0` |
| GGUF 繁中夾簡體 | 無 beam，逐 token 選字 | 輸出後以 OpenCC `s2twp` 後處理 |
| 長文尾段不完整 | 超過 context 或 `max_new_tokens` 不足 | 分段翻譯後串接 |

完整接入說明、GGUF 與 transformers 路徑差異，以及可交給外部工程師／agent 的檢查清單，見 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。解碼搜尋腳本與結果：`scripts/decode_search.py`、`results/decode_search/`。

### 提示詞格式

與訓練資料一致：

```
system: You are a professional translator.
user:   翻譯成繁體中文：\n{原文}
        # 或「翻譯成英文：」「翻譯成日文：」
```

載入類別請使用 `AutoModelForImageTextToText`（transformers ≥ 5.x），並以 chat template 組 prompt；不宜當作一般 CausalLM 手拼字串。

## 環境需求

- Windows 11 / Python 3.12 / [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU ≥ 8GB VRAM（CUDA 12.8）

```powershell
uv sync                          # 主環境：訓練、推論、評測
uv sync --project tools/comet    # COMET 評分隔離環境
```

`unbabel-comet` 會鎖定較舊的 transformers，與 Qwen3.5 所需的 5.x 衝突，故獨立為子專案。Windows 上需安裝 `triton-windows` 與 `flash-linear-attention`，否則線性注意力退回較慢的 torch 路徑，較大 batch 時容易顯存不足。

## 重現訓練與評測

```powershell
uv run python scripts/download_data.py
uv run python scripts/prepare_data.py --limit 80000 --dev-from data/sft/dev.jsonl
uv run python scripts/train_sft.py --config configs/sft_lora_v5e.yaml
uv run python scripts/evaluate.py --tag v5e --adapter outputs/sft-v5e --full
uv run --project tools/comet python tools/comet/score.py --tag v5e-flores
uv run python scripts/eval_bench.py      --tag v5e --adapter outputs/sft-v5e
uv run python scripts/eval_capability.py --tag v5e --adapter outputs/sft-v5e --axis all
uv run python scripts/regression_guard.py --candidate v5e
```

## 品質門檻（機器判定）

`scripts/regression_guard.py` 將六類門檻一次跑完；結束碼 **0 = 通過、1 = 未通過、2 = 缺值**（未跑過的項目不視為通過）。

| 項目 | 門檻 |
|---|---|
| 簡體洩漏 en→zhtw / ja→zhtw | ≤ 官方基底 + 0.3 個百分點 |
| 六方向 COMET | 各方向不得**顯著**低於官方基底（見下） |
| 翻譯機率（通用題被當翻譯作答） | ≤ 5.0% |
| BELEBELE / 知識 三語 | 各 ≥ 官方基底 − 3.0 |
| 長文尾段譯出比 `tail_ratio_median` | ≥ 0.80（六方向） |
| 長文腰斬率 `truncated_pct` | ≤ 5%（六方向） |

COMET 採三段判定：≥ 官方 → 通過；落在 `[官方−0.5, 官方)` → 以 `tools/comet/paired_bootstrap.py` 的 95% CI 是否跨 0 決定；< 官方−0.5 → 未通過。理由是 n≈1000 的 system score 約有 ±0.4 的雜訊，不宜僅用小數點後兩位的點估計判勝負。

v5e 在上述門檻下 **28 格全數通過**。較大 LoRA（r=128，v5f）COMET 略高，但 BELEBELE 中日文明顯退步，故未作為發布版。

## 專案結構

大型產物（語料、權重、GGUF）不納入版控，見下方「倉庫未包含的內容」。

```
├─ README.md                  本文件
├─ CLAUDE.md / AGENTS.md      協作者與 AI agent 工作規範
├─ CONTEXT.md                 開發交接摘要
├─ LICENSE                    Apache-2.0
├─ pyproject.toml / uv.lock
│
├─ docs/
│    ├─ REPORT.md             研究報告（版本沿革、方法、結論）
│    ├─ RESEARCH-v5.md        v5 實驗原始紀錄
│    ├─ INTEGRATION.md        應用接入、解碼參數、常見問題
│    └─ assets/loss_curve.png
│
├─ configs/                   訓練設定（每版獨立檔案）
│    ├─ sft_lora_v5e.yaml     發布版：r64/α128、1 epoch、約 503k 筆
│    └─ …
│
├─ scripts/                   資料 → 訓練 → 評測 → 匯出
│    ├─ download_data.py / prepare_data.py / train_sft.py
│    ├─ evaluate.py           六方向翻譯評測；解碼常數定義處
│    ├─ decode_search.py      解碼參數搜尋
│    ├─ eval_capability.py / eval_bench.py / eval_gguf.py
│    ├─ regression_guard.py   品質門檻機器判定
│    └─ export_model.py / export_gguf.py
│
├─ tools/comet/               COMET 隔離子專案
├─ results/                   評測結果與對照基準（納入版控）
├─ tasks/                     待辦與教訓紀錄
├─ data/ outputs/ release/ logs/   （gitignored）
```

### 倉庫未包含的內容

| 路徑 | 約略大小 | 取得方式 |
|---|---|---|
| `data/` | ~2 GB | `download_data.py` + `prepare_data.py`（語料授權各異，不 re-host） |
| `outputs/` | ~10 GB | 重新訓練；v5e adapter 目錄為 `outputs/sft-v5e` |
| `release/` | ~4.5 GB | 與 HF 模型庫對應的本機上傳鏡像 |
| `results/hyp/` | ~35 MB | 執行 `evaluate.py` 產生 |
| `logs/` | ~3 MB | 執行日誌 |

## 訓練資料

v5e：六方向各約 80,000 句，合計 **502,993** 句，外加通用指令 replay 35,177 筆（dev 1,200 跨版凍結）。各來源設佔比上限與領域標籤，避免單一語料壟斷。

| 來源 | 用途 | 說明 |
|---|---|---|
| COCT（Taiwan Panorama） | en↔zhtw | 原生臺灣正體 |
| TED2020 | en↔zhtw、ja↔zhtw、en↔ja | 演講字幕 |
| WikiMatrix / JParaCrawl / KFTT / Tatoeba / News-Commentary | en↔ja | 書面體平行語料 |
| GlobalVoices / KDE4 | en↔zhtw、ja↔zhtw | 新聞與軟體在地化 |
| OpenSubtitles / MTNT | 僅 →en 的源側 | 補強英譯，降低對洩漏的副作用 |
| OPUS-100 | en↔zhtw | 量大、雜訊較多 |
| oasst2 / aya_dataset | replay | 通用指令，減輕災難性遺忘 |

清洗流程摘要：正規化與長度過濾 → 語言驗證 → **LaBSE 雙語語意過濾（≥0.65）** → OpenCC `s2twp` 統一臺灣正體 → 去重 → **評測集污染閘**（與五個評測基準的語言行 hash 比對，命中即丟）→ 來源配額。統計見 `results/data_stats.json`。

## 訓練設定

LoRA 目標模組涵蓋標準注意力、MLP，以及 Qwen3.5 的**線性注意力層**（`in_proj_qkv/z/a/b`、`out_proj`）。bf16、不使用 packing、以 token 預算組 batch、`max_length` 1408、學習率 1e-4 cosine、單 epoch。

| 設定檔 | 說明 | 產出 |
|---|---|---|
| **`sft_lora_v5e.yaml`** | r64/α128、1 epoch、約 503k 筆 | `outputs/sft-v5e`（**發布版**） |
| `sft_lora_v5f.yaml` | r128/α256（COMET 略高但踩品質門檻） | `outputs/sft-v5f` |

v5e 實測成本（RTX 5060 Ti 16GB）：峰值 VRAM **4.00GB**、約 **10.5 小時**、5,677 步、final eval_loss **1.7708**。

![v5e SFT loss](docs/assets/loss_curve.png)

> **顯存注意**：Qwen3.5 詞表約 248K，logits 隨 batch×seq 膨脹。Windows 上超顯存時驅動可能靜默改走系統記憶體而非 OOM，速度會驟降。長跑前可用 `scripts/bench_step.py` 量測。僅需最後一格 logits 的 forward 應設 `logits_to_keep=1`。

## 發布產物

程式碼與實驗證據在 GitHub；權重與模型卡在 Hugging Face（可視權限設定為私人或公開）。

| 位置 | 內容 |
|---|---|
| [Hugging Face](https://huggingface.co/RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja) | LoRA、合併 bf16、GGUF、[模型卡](https://huggingface.co/RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja/blob/main/README.md) |
| [GitHub](https://github.com/RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja) | 訓練／評測程式、設定、`results/`、報告 |
| 本機 `release/` | 與 HF 模型庫同結構的上傳鏡像 |

```
release/
├─ README.md                     模型卡（含 YAML frontmatter）
├─ adapter_model.safetensors     LoRA r64/α128（約 173MB）
├─ adapter_config.json
├─ tokenizer.* / chat_template.jinja
├─ merged-bf16-v5e/              合併全模型約 1.7GB
└─ gguf-v5e/                     Q8_0 / Q4_K_M / f16
```

```powershell
uv run python scripts/export_model.py --adapter outputs/sft-v5e --out release/merged-bf16-v5e
uv run python scripts/export_gguf.py --llama-cpp <llama.cpp> --quantize-bin <llama-quantize>
hf upload <repo-id> release/ .
```

GGUF 實測（RTX 5060 Ti，`-ngl 99`）：Q8_0 約 **186 t/s**，Q4_K_M 約 171–217 t/s。使用 llama-cli 時應加上 `--reasoning off --reasoning-budget 0`；CJK 提示詞建議以 `-f prompt.txt`（UTF-8）餵入，避免 Windows 主控台編碼問題。GGUF 路徑為 greedy，建議對繁中輸出做 OpenCC `s2twp`。轉檔若遇 `blk.24` 缺失，主檔轉換需加 `--no-mtp`。細節見 HF 模型卡與 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。

## 授權

模型與程式碼依 **Apache-2.0**。各語料授權依其原始來源；本專案僅發布微調權重，不轉發語料。
