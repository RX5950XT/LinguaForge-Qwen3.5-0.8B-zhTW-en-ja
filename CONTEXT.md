# CONTEXT — 開發紀錄交接

給下一個接手的 agent。**規範看 `CLAUDE.md`（含出貨標準），這裡只記狀態與判斷依據。**
完整實驗過程在 `docs/RESEARCH-v4.md` / `docs/RESEARCH-v5.md`，踩坑在 `tasks/lessons.md`，
進度勾選在 `tasks/todo.md`。

## 專案目標

把 `Qwen/Qwen3.5-0.8B`（873M，Apache-2.0）LoRA SFT 成 zh-TW↔en↔ja 六方向翻譯特化模型。
純開源語料、本機單卡（RTX 5060 Ti 16GB）。

**倉庫維持私人（GitHub + HF 皆是，未經明確指示不得轉公開）。**

## 🚨 最新狀態（2026-07-31 03:0x）

**目前最佳 = `outputs/sft-v5f`**（LoRA r=128 / alpha=256，資料同 v5e，1 epoch 5,677 步）。

| 指標 | base | v5d | v5e | **v5f** |
|---|---|---|---|---|
| COMET AVG（FLORES 六方向） | 84.59 | 86.34 | 86.56 | **86.74** |
| eval_loss（凍結 dev） | — | 1.831 | 1.7708 | **1.7440** |
| 通用能力 n=90 | 78.9 | 71.1 | 72.2 | **73.3** |
| 翻譯機率 | 1.1 | 2.2 | 3.3 | 3.3 |
| BELEBELE 三語均（輪轉去偏） | 55.14 | — | 57.48 | 跑中 |
| 知識三語均（TMMLU+/MMLU/MMMLU-JA） | 36.81 | — | 36.99 | 跑中 |
| 訓練耗時 / 峰值 VRAM | — | — | 10h30m / 4.00GB | 11h12m / 4.77GB |

六方向 COMET（v5f）：en→zhtw 86.34、zhtw→en 85.35、en→ja 88.79、ja→en 86.39、
ja→zhtw 86.23、zhtw→ja 87.31。`regression_guard.py --candidate v5f --baseline v5e` = PASS。

### 三個已定案的結論

1. **純加資料的路走完了**（F41）。×1.77 換 +0.25 COMET、×1.84 換 +0.23，報酬持平到遞減，
   且 ja↔zhtw 已觸池底（75,105/80,000，文件級只有 7,105）。
2. **容量才是當前瓶頸**（F42）。r 64→128 用 +6.7% 時間、+0.77GB 換 +0.18 COMET，
   性價比比加資料高一個量級。連 en→zhtw 這個被資料量 ×3.6 推不動的方向都解凍了
   （86.23 兩版分毫不差 → 86.34，chrF++ / BLEU 同步上升）。
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

只剩 **r=256** 一個可跑的實驗（配 `configs/sft_lora_v5g.yaml`，仿 v5f 只改 lora.r/alpha）。
照停止規則：v5d→v5e +0.23、v5e→v5f +0.18 都 ≥0.10，還沒觸發收工；
若 v5g < 0.10 就是連續第一次，再一版就停。

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
