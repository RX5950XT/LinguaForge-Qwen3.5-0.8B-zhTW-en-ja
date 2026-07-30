# CONTEXT — 開發紀錄交接

給下一個接手的 agent。**規範看 `CLAUDE.md`（含出貨標準），這裡只記狀態與判斷依據。**
完整實驗過程在 `docs/RESEARCH-v4.md` / `docs/RESEARCH-v5.md`，踩坑在 `tasks/lessons.md`，
進度勾選在 `tasks/todo.md`。

## 專案目標

把 `Qwen/Qwen3.5-0.8B`（873M，Apache-2.0）LoRA SFT 成 zh-TW↔en↔ja 六方向翻譯特化模型。
純開源語料、本機單卡（RTX 5060 Ti 16GB）。

**倉庫維持私人（GitHub + HF 皆是，未經明確指示不得轉公開）。**

## 🚨 最新狀態（2026-07-31 03:3x）

**出貨候選 = `outputs/sft-v5e`。v5f 翻譯更好但踩到硬閘，不得出貨。**

v5f（r=128）COMET AVG 比 v5e 高 0.18，但 BELEBELE 中日文分別掉 4.53 / 7.84
（門檻是 ≥ base − 3.0）→ **容量翻倍把中日文的通用理解洗掉了，換到英文與翻譯分數**。

| 指標 | base | v5d | v5e | **v5f** |
|---|---|---|---|---|
| LoRA r / 資料筆數 | — | 64 / 273k | 64 / 503k | **128 / 503k** |
| COMET AVG（FLORES 六方向） | 84.59 | 86.34 | 86.56 | **86.74** |
| eval_loss（凍結 dev） | — | 1.831 | 1.7708 | **1.7440** |
| 通用能力 n=90 | 78.9 | 71.1 | 72.2 | **73.3** |
| 翻譯機率 | 1.1 | 2.2 | 3.3 | 3.3 |
| BELEBELE 三語均（輪轉去偏） | 55.14 | **58.55** | 57.48 | 53.47 ❌ |
| └ zh-TW / ja / en | 55.81/51.78/57.83 | 57.25/50.31/68.08 | 57.28/52.36/62.81 | 51.28/43.94/65.19 |
| 知識三語均（TMMLU+/MMLU/MMMLU-JA） | 36.81 | **37.78** | 36.99 | 35.25 |
| └ zh-TW / ja / en | 30.67/36.08/43.67 | 31.06/36.47/45.81 | 30.53/36.64/43.81 | 29.33/34.75/41.67 |
| 訓練耗時 / 峰值 VRAM | — | — | 10h30m / 4.00GB | 11h12m / 4.77GB |

六方向 COMET（v5f）：en→zhtw 86.34、zhtw→en 85.35、en→ja 88.79、ja→en 86.39、
ja→zhtw 86.23、zhtw→ja 87.31。`regression_guard.py --candidate v5f --baseline v5e` = PASS。

### 三個已定案的結論

1. **純加資料的路走完了**（F41）。×1.77 換 +0.25 COMET、×1.84 換 +0.23，報酬持平到遞減，
   且 ja↔zhtw 已觸池底（75,105/80,000，文件級只有 7,105）。
2. **r=128 是斷崖，不是斜坡**（F42/F48/F49）。三點連線後很乾淨：
   **r=64 的兩版（v5d/v5e）兩軸都 ≥ base，資料量 ×1.84 不影響通用能力；
   只有 r=128 崩**（BELEBELE 中日文 −4.53 / −7.84，門檻 base−3.0）。
   所以**退化的變因是 r，不是資料量、不是訓練步數**。
   機制看得到：v5f 在 zh-TW/ja 押同一個選項 55.2% / 56.9%（英文 28.6%，接近均勻）
   ——**中日文已經退回固定先驗，不太在作答**。自建 n=90 面板完全沒抓到（還顯示上升），
   因為那批題目的形式離訓練資料太近。**這就是引入外部基準的價值。**
   代價換算：v5e→v5f 用 −4.01 BELEBELE 換 +0.18 COMET，不划算。
3. **replay 比例不是變因，絕對量才是**（F40）。比例 22.9%→12.9%→7.0% 一路稀釋，
   絕對量固定 35,177 筆，通用能力反而 70.0→71.1→72.2 上升。

### 未解問題

- **通用能力比 base 低 5.6**（73.3 vs 78.9），翻譯機率 3.3% vs base 1.1%。
  要補得擴充 replay 池，但 `build_replay.py` 三個來源（oasst2 / aya / oasst2-33k-ja）已抽乾，
  `REPLAY_SHARE = 0.35` 這個設計目標從 v4 起就沒滿足過。**卡在找不到 Apache-2.0 相容、
  非機翻的通用指令語料。**
- **en→zhtw 仍是最弱環節**：vs base 只有 +0.15，其餘五向 +0.6~+3.5。
  原本的 F38 假設（目標側語域是字幕、非書面語）**證據不足**——容量一加就動了，
  比較像是 r=64 撐不起六方向。
- **`--axis doc` / `ifeval` 兩軸從 v4 後就沒跑過**。v2/v3 的「長文只翻前兩段」失敗模式，
  整個 v5 系列沒有任何一版驗證過。**這是現在最大的盲區。**

### 下一步

**r=256 取消，不是暫緩**（F46）。三點連線顯示 r=128 已經越線，r=256 只會更深，
而修正手段（擴充 replay）卡在授權牆。可用的容量區間是 **r ≤ 64**。

1. **補 doc / ifeval 兩軸**（F47，跑中）：v5e 與 base 各 25 篇，這是最後一道沒量過的硬閘
2. 若 v5e 過 doc 閘 → **具備出貨資格**，可打包（`release/` 還停在 v3）
3. 想再往上：r=96 是唯一沒探過的中間點，但先確認 doc 軸沒事再說

停止規則現況：COMET AVG 的 v5d→v5e +0.23、v5e→v5f +0.18 都 ≥0.10，尚未觸發收工；
但硬閘已經攔下 v5f，且 r 這條路到頂 —— **實質上只剩資料來源（授權牆）可動**。

## 評測堆疊

| 層 | 腳本 | 產出 |
|---|---|---|
| 翻譯品質 | `evaluate.py`（flores/ntrex/wmt22/alt/tico19） | `results/<版本>/<tag>-<bench>.json` |
| 語意 | `tools/comet/score.py --tag <tag>-flores` | 併回同一個 json |
| 行為退化 | `eval_capability.py --axis doc\|ifeval\|general` | `results/capability/<tag>.json` |
| 通用能力 | `eval_bench.py`（BELEBELE + TMMLU+/MMLU/MMMLU-JA） | `results/bench/<tag>.json` |
| 護欄 | `regression_guard.py --candidate X --baseline Y` | exit 1 = 不得出貨 |

- COMET 的 `--tag` 要**全名**（`v5e-flores`），不是版本號
- `eval_bench.py` 預設 `--scoring rotate`（輪轉去偏，正式數字）；`letter` 只是診斷，
  兩者分數不可互比，存不同檔名
- base 的 n=90 通用能力在 `results/capability/base-n90.json`（**不是** `base.json`，
  那支只有 n=12 的舊資料，別拿來當對照）

## 操作紀律

- **背景訓練會隨 Claude Code session 一起死**。Windows detach 三種方法全失敗
  （Intel Fortran runtime 的 `CTRL_CLOSE_EVENT` handler），**不要再試第四種**。
  對策是 `save_steps: 100`（≈11 分鐘），斷了最多重跑這麼多
- **resume 時 CLI/config 參數會被 `<checkpoint>/trainer_state.json` 蓋掉**，只有一行警告。
  要改就直接改 checkpoint 裡的 json；驗證方式是「下個 checkpoint 有沒有落在新間隔」
- 訓練 log 的 loss 要等 process 結束才 flush，中途看 `checkpoint-*/trainer_state.json`
- Windows 超過 VRAM **不會 OOM**，驅動靜默 fallback 到系統記憶體，速度掉到 1/5 且無警告。
  `nvidia-smi` 滿載但吞吐異常低 → 用 `bench_step.py` 量實際 VRAM

## 目錄結構

```
configs/   每版一個訓練 config（sft_lora / _v3 / _v5f / sft_qlora_2b），不改舊檔
scripts/   資料管線 + 訓練 + 評測；test_*.py 是不需 GPU 的自檢
tools/comet/  COMET 隔離子專案（獨立 uv 環境，鎖 transformers<4.58，絕不可併入主環境）
data/      raw/（21 份 tsv）、sft/（train/dev/replay.jsonl）、flores200/、eval_lines.txt（污染閘）
outputs/   各版 adapter；舊版只留最終 adapter，checkpoint-* 已清
results/   依版本分類，慣例見 results/README.md（讀取端一律 rglob）
logs/      依類型分：data/ bench/ train/ eval/ comet/ export/
tasks/     todo.md（進度）、lessons.md（教訓）
docs/      REPORT.md、RESEARCH-v4.md、RESEARCH-v5.md
release/   (gitignored) HF repo 鏡像；**目前仍是 v3，落後主線很多**
```

## 待辦（非阻塞）

- `release/` 還在 v3，模型卡未更新，每目標語言 `DECODE` 預設沒寫進去
- `docs/RESEARCH-v4.md` 還欠 v4 結果
- v5a 能力面板不完整；F28（base 用 `--full` + 現行 `DECODE` 重跑）沒做
- `token_budget: 1450` 是舊 3070 Ti 8GB 調的；16GB 卡要調高得同步降
  `gradient_accumulation_steps`，並先用 `bench_step.py` 實測，**不可混進實驗當混淆變因**
