# CONTEXT — 開發紀錄交接

給下一個接手的 agent。**規範看 `CLAUDE.md`（含出貨標準），完整實驗史看 `docs/REPORT.md`，
踩坑通則看 `tasks/lessons.md`，未解項看 `tasks/todo.md`。這裡只記「現在是什麼狀態」。**

## 專案目標與現況

把 `Qwen/Qwen3.5-0.8B`（873M，Apache-2.0）LoRA SFT 成 zh-TW↔en↔ja 六方向翻譯特化模型。
純開源語料、本機單卡（RTX 5060 Ti 16GB）。

**🚨 倉庫維持私人（GitHub + HF 皆是，未經明確指示不得轉公開、不得上傳）。**

**狀態：訓練線已收工（2026-07-31）。出貨版 = `outputs/sft-v5e`，已打包 `release/`，
已推 GitHub（code）與 HF（權重，私人）。**

## 為什麼收工

`regression_guard.py --candidate v5e` → **exit 0，28 格全過**（其中 en→zhtw 靠 CI 判平手）。
而五個可動的變因全部結案：

| 槓桿 | 狀態 | 依據 |
|---|---|---|
| 資料量 `--limit` | 走完 | ×1.77→+0.25、×1.84→+0.23；ja↔zhtw 已抽不滿（75,105/80,000） |
| LoRA r | 上界 r ≤ 64 | r=128 的 BELEBELE 中日文 −4.53/−7.84，**斷崖不是斜坡** |
| replay 擴充 | 卡授權 | 三來源抽乾；Apache-2.0 相容的非機翻 zh-TW 指令集找不到 |
| 解碼 | 已收割 | +0.92 COMET，零訓練成本 |
| epoch | 已證偽 | v5c 兩 epoch/141k（1.866）輸 v5d 一 epoch/273k（1.831） |

唯一沒探過的 r=96，樂觀線性內插也只值 +0.08~0.12 COMET（→ 86.64~86.68），
**離目標 87.00 還差 0.32 以上**，而 r 是斷崖所以連內插前提都不成立。
→ **要再往上必須先有新語料，不是先有新訓練。**

## 四版對照（FLORES 全量 n=1012、beam4 + 逐語言 nrng）

| 指標 | base | v5d | **v5e（出貨）** | v5f |
|---|---|---|---|---|
| LoRA r / 資料筆數 | — | 64 / 273k | **64 / 503k** | 128 / 503k |
| COMET AVG | 84.64 | 86.34 | **86.56** | 86.73 |
| eval_loss（凍結 dev） | — | 1.831 | **1.7708** | 1.7440 |
| BELEBELE zh-TW / ja / en | 55.81/51.78/57.83 | 57.25/50.31/68.08 | **57.28/52.36/62.81** | 51.28/43.94/65.19 ❌ |
| 知識 zh-TW / ja / en | 30.67/36.08/43.67 | 31.06/36.47/45.81 | 30.53/36.64/43.81 | 29.33/34.75/41.67 |
| 自建通用 n=90 / 翻譯機率 | 78.9 / 1.1 | 71.1 / 2.2 | 72.2 / 3.3 | 73.3 / 3.3 |
| 硬閘 | — | exit 2（doc 未量） | **exit 0** | exit 1 |

六方向 COMET：en→zhtw **86.32 → 86.23 是統計平手**（CI [−0.414, +0.225] 跨 0），
其餘五向 +0.48 ~ +4.66。洩漏 en→zhtw 10.18%→1.09%、ja→zhtw 43.58%→0.69%。

> **v5f 翻譯更好但踩硬閘，不得出貨**：容量翻倍把中日文的通用理解洗掉了
> （在 zh-TW/ja 押同一選項 55.2%/56.9%，英文 28.6%）。自建 n=90 面板完全沒抓到，
> **這就是引入外部基準的價值**。

## 評測堆疊

| 層 | 腳本 | 產出 |
|---|---|---|
| 翻譯品質 | `evaluate.py`（flores/ntrex/wmt22/alt/tico19） | `results/<版本>/<tag>-<bench>.json` |
| 語意 | `tools/comet/score.py --tag <tag>-flores` | 併回同一個 json |
| 顯著性 | `tools/comet/paired_bootstrap.py --a A --b B --direction d` | `results/bootstrap/*.json` |
| 行為退化 | `eval_capability.py --axis doc\|ifeval\|general` | `results/capability/<tag>.json` |
| 通用能力 | `eval_bench.py`（BELEBELE + TMMLU+/MMLU/MMMLU-JA） | `results/bench/<tag>.json` |
| 護欄 | `regression_guard.py --candidate X` | exit 0=過／1=FAIL／**2=缺值** |

- **護欄是唯一的出貨判準**，六條硬閘全在裡面，不要再人工核對表格。
  COMET 落在 `[base−0.5, base)` 時它會去 `results/bootstrap/` 找 CI 裁決，找不到判 MISSING
- COMET 對照**只認** `results/baseline/base-full-flores.json`（n=1012 + 現行 `DECODE`）。
  `base-b4-flores.json` 的 comet 欄位全是 null，已作廢
- COMET 的 `--tag` 要**全名**（`v5e-flores`），不是版本號
- `eval_bench.py` 預設 `--scoring rotate`（輪轉去偏，正式數字）；`letter` 只是診斷，
  兩者分數不可互比，存不同檔名
- base 的 n=90 通用能力在 `results/capability/base-n90.json`（**不是** `base.json`，那支是 n=12 舊資料）

## 操作紀律

- **背景訓練會隨 Claude Code session 一起死**。Windows detach 三種方法全失敗
  （Intel Fortran runtime 的 `CTRL_CLOSE_EVENT` handler），**不要再試第四種**。
  對策是 `save_steps: 100`（≈11 分鐘）＋ `--resume-from-checkpoint auto`
- **resume 時 CLI/config 參數會被 `<checkpoint>/trainer_state.json` 蓋掉**，只有一行警告。
  要改就直接改 checkpoint 裡的 json；驗證方式是「下個 checkpoint 有沒有落在新間隔」
- 訓練 log 的 loss 要等 process 結束才 flush，中途看 `checkpoint-*/trainer_state.json`
- Windows 超過 VRAM **不會 OOM**，驅動靜默 fallback 到系統記憶體，速度掉到 1/5 且無警告。
  `nvidia-smi` 滿載但吞吐異常低 → 用 `bench_step.py` 量實際 VRAM
- `prepare_data.py` 的 `--limit` **必給**（內建預算是 130,000/方向）；跑完先核對
  `results/data_stats.json` 的 `directions` 跟上一版對不對得上，再開訓

## 目錄結構

```
configs/   每版一個訓練 config，不改舊檔；出貨版凍結在 sft_lora_v5e.yaml
scripts/   資料管線 + 訓練 + 評測；test_*.py 是不需 GPU 的自檢
tools/comet/  COMET 隔離子專案（獨立 uv 環境，鎖 transformers<4.58，絕不可併入主環境）
data/      raw/、sft/（train 502,993 / dev 1,200 凍結）、flores200/、labse/、eval_lines.txt
outputs/   各版 adapter；舊版只留最終 adapter，checkpoint-* 已清
results/   依版本分類，慣例見 results/README.md（讀取端一律 rglob）
logs/      依類型分：data/ bench/ train/ eval/ comet/ export/
tasks/     todo.md（現況與未解項）、lessons.md（教訓）
docs/      REPORT.md（v1→v5e 全紀錄）、RESEARCH-v5.md（逐項發現原始紀錄）
release/   (gitignored) HF repo 鏡像，**只留 v5e**（根目錄 adapter + merged-bf16-v5e/
           + gguf-v5e/ + 模型卡 README.md + assets/）
```

## 連結

| | |
|---|---|
| GitHub（程式，私人） | https://github.com/RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja |
| HF（權重 v5e，私人） | https://huggingface.co/RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja |

## 下一步

1. 想再往上只剩「找到新的 Apache-2.0 相容 zh-TW 書面語／通用指令語料」一條路。
   **有語料才有下一版訓練**，反過來不成立。
2. 非阻塞待辦見 `tasks/todo.md`。
3. 倉庫維持私人；**未經明確指示不得轉公開**。
