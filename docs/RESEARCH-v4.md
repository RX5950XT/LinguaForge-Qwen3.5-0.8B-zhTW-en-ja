# v4 研究：翻譯特化為什麼把模型做壞了，以及該怎麼修

> 2026-07-27。起因：使用者手測 v3 覺得「怪怪的」，實測確認災難性遺忘。
> 本文記錄根因、文獻調查、修法與新評測設計。舊實驗紀錄見 `REPORT.md`。

## 1. 事故摘要

v2/v3 在五個基準（FLORES / NTREX / WMT22 / ALT / TICO-19）上 COMET 84~86，
帳面上「翻譯品質贏過 baseline」。但手動測試發現三類基準完全測不到的缺陷：

| 缺陷 | 表現 | 嚴重度 |
|---|---|---|
| **通用能力歸零** | 任何非翻譯指令都被當成待翻譯文本，直接把問題翻掉 | 致命 |
| **長篇漏譯** | 5 段 1491 字的文章只翻前 2 段就發 `<im_end>` 收尾 | 致命 |
| **無限重複** | v2 的 zhtw→ja 對長文輸出同一句 30+ 次（1872 字） | 致命 |
| 標點消失 | 34% zhtw / 39% ja 訓練目標句無句末標點 → 模型學會不加標點 | 中 |
| 符號旁插空格 | `API_ KEY`、`https://example.com/ zh- tw? ref= news` | 中 |

實測（8 題通用能力，貪婪解碼）：

| 問題 | base | v3 |
|---|---|---|
| 台灣最高山？海拔？ | 玉山，3952 公尺 | 「臺灣最高的山是哪一座？」← 把問題翻成繁中 |
| 星期三＋3 天？ | 星期五（理由錯但有答） | `If today is Wednesday, then three days later it will be:` |
| 1200 打七折再折 100 | 740 元 ✅ | 把題目翻成英文 |
| 寫 Python 偶數加總 | 完整正確程式碼 | 把需求翻成英文 |
| 列三種水果用頓號 | 香蕉 蘋果 葡萄 | 把指令翻成英文 |

## 2. 根因

三者疊加，**第一項是主因**：

1. **訓練資料 100% 翻譯任務、0% 通用資料。**
   `CONTEXT.md`「已確認決策」原文寫「純翻譯特化（不混通用資料）」。
   結果模型學到的不是「翻譯這個任務」，而是「chat 模板下的 user 內容一律翻譯」——
   指令跟隨的先驗被整個覆蓋掉。

2. **訓練資料 100% 單句對、0% 多段落樣本。**
   77 萬筆全部是句級平行語料，模型學會「一句進、一句出、發 `<im_end>`」。
   遇到多段輸入就在第一二句後收尾。

3. **資料噪音沒清乾淨。** TED2020/OpenSubtitles 無標點、KDE4 tokenized 未 detok。

不是 LoRA rank 的問題：v3（r64/α128/1ep/neftune）與 v2（r32/α64/2ep）崩得一樣。

## 3. 文獻調查

### 3.1 混多少通用資料？——SOTA 的答案是「通用遠多於翻譯」

**Tower+（Unbabel, 2025）** 是目前翻譯特化 LLM 裡唯一明確處理「翻譯 vs 指令跟隨」
權衡並公布配方的工作。其四階段後訓練的資料配比：

| 階段 | 配比 |
|---|---|
| 持續預訓練 CPT | 66% 單語 / 33% 平行 / **1% 指令資料** |
| **監督微調 SFT** | **約 22% 翻譯 / 約 78% 通用指令**（1.3M 樣本） |
| 偏好優化 WPO | SFT prompts + UltraFeedback |
| RLVR | Tülu 3 + 翻譯專屬 verifiable reward |

SFT 的通用資料來源：OpenHermes-2.5、Aya、Daring-Anteater、Magpie、Tülu，
涵蓋數學、程式、問答；翻譯側則來自 WMT 與 FLORES。

論文明確記載的發現：**CPT 階段提升翻譯品質，但代價是通用能力一致性退化**；
後續的 WPO 階段才同時把指令跟隨、通用對話、翻譯三者一起拉回來。

→ **使用者的直覺（通用資料要比翻譯資料更多）與 SOTA 一致。本專案的 100:0 是極端錯誤配比。**

### 3.2 replay 比例的一般結論

- 只 replay **1%** 歷史指令資料（CT0 作法）就足以維持跨任務的指令跟隨能力，記憶體開銷極小。
- 在 finetuning 時混入通用預訓練資料不只防遺忘，**還能提升目標任務表現**，
  資料效率最高可達 2.06×。
- 專精化之前先做通用指令微調，可隨模型規模增長緩解遺忘。

→ 本專案 0.8B 模型 + LoRA，建議通用 replay 佔比取 **30~50%**（比 Tower+ 的 78% 保守，
因為我們只要守住 zh-TW / ja / en 三語，且 GPU 只有 8GB）。

### 3.3 長篇翻譯要用文件級資料訓

DocBlocks（2025）等工作的結論一致：**句級微調的模型在文件級評測上不成立**，
必須用整份文件與帶上下文的片段來訓，並同時提供「有上下文」與「無上下文」兩種指令，
模型才學得到跨句依賴。本專案的 TED2020 / GlobalVoices / KDE4 原本就有文件邊界，
可以把已清洗的句對按原文件重新拼回段落，不需要新語料。

## 4. 修法（v4 資料配方）

| 問題 | 修法 | 預估成本 |
|---|---|---|
| 通用能力歸零 | 混入 30~50% 通用指令資料（三語） | 資料準備 1~2h |
| 長篇漏譯 | 15~20% 的翻譯樣本改成文件級（多段落），由現有語料按文件邊界重拼 | 腳本 1h |
| 標點消失 | 砍「源有句末標點但譯文無」的不對稱樣本（79,767 筆 / 10.3%） | 過濾即可 |
| 符號旁空格 | 砍命中殘留空格模式的樣本（11,862 筆 / 1.5%） | 過濾即可 |

### 4.1 通用資料來源（已實作於 `scripts/build_replay.py`）

| 資料集 | 授權 | 採用 | 實得 |
|---|---|---|---|
| `OpenAssistant/oasst2` | **Apache-2.0** | ✅ 人工撰寫對話樹，取 rank 最佳助理回覆 | en 14.0K / zh 1.9K / ja 0.2K |
| `CohereLabs/aya_dataset` | **Apache-2.0** | ✅ 人工標註 prompt-completion | ja 6.3K / en 3.9K |
| `llm-jp/oasst2-33k-ja` | **Apache-2.0** | ✅ 日文量體補充 | ja 56.6K（與 aya 大量重複） |
| `CohereLabs/aya_collection`（traditional_chinese）| Apache-2.0 | ❌ **機翻 Flan，品質崩壞** | — |
| `yentinglin/TaiwanChat` | CC-BY-NC-4.0 | ❌ 與專案 Apache-2.0 衝突 | — |
| `allenai/tulu-3-sft-mixture` | ODC-BY | ❌ 部分子集非商用 | — |

**aya_collection 的 `traditional_chinese` 有 3.6M 筆看似解決缺口，實測不可用**：
它是英文 Flan 的機器翻譯，連編號都被當文字翻譯——`1.` 變成「一,他們」、`2.` 變成
「2. 我們的國家」、`No.` 變成「沒有任何問題.」。混進訓練會直接毒化輸出格式。

去重清洗後實得（`data/sft/replay.jsonl`，35,083 筆）：

| 語言 | 筆數 |
|---|---|
| en | 16,579 |
| ja | 17,157 |
| **zhtw** | **1,347** ⚠ |

**繁中仍是缺口**（僅 en/ja 的 8%）。公開語料裡沒有授權乾淨、非機翻、足量的繁中指令集。
補法（需 GPU，成本待實測）：以官方 `Qwen/Qwen3.5-2B`（Apache-2.0）為 teacher，
把英文 replay 的 prompt+answer 翻成 zh-TW 或直接生成 zh-TW 回答，輸出過 s2twp。
這組合本專案已驗證過（COMET 88.21、洩漏 0.2%），拿來當 teacher 是現成且乾淨的選擇。

## 5. 新評測設計（`scripts/eval_capability.py`）

舊 `evaluate.py` 的五個基準**全是單句、全是翻譯任務**，所以上述致命缺陷一個都沒被抓到。
COMET 84.8 是真的，但它只涵蓋「單句翻譯」這一個 use case。新面板補三軸：

### 軸 A：文件級翻譯（`--axis doc`）

資料源 **WMT24++**（`google/wmt24pp`，Apache-2.0）。選它的理由：

- **明確包含 `en-zh_TW`**，且論文記載對繁中做過 native speaker 抽樣檢查
  （不是 zh_CN 轉繁），這在公開基準裡很少見
- 有 `document_id` / `segment_id` → 可聚合成文件級
- `en-zh_TW` 與 `en-ja_JP` 共用同一批英文原文 → **en-pivot 得到全六方向多平行**
- 四領域：news / social / speech / literary（FLORES 只有維基書面體）
- 每語對 998 段，聚合後約 70 篇文件、平均 14 句/篇

指標（重點是完整度，不是只有 chrF++）：

| 指標 | 意義 |
|---|---|
| `chrf++` | 傳統品質 |
| `completeness_median` | 譯文長度 / 參考長度的中位數，<1 代表系統性漏譯 |
| `truncated_pct` | 長度比 <0.5 的比例（腰斬＝漏譯） |
| `bloated_pct` | 長度比 >2.0 的比例（膨脹＝重複） |
| `repetitive_pct` | 字元 8-gram 重複率 >0.3 的比例（抓無限複述） |
| `simplified_leak_pct` | 簡體洩漏（沿用舊指標） |

### 軸 B：可驗證指令跟隨（`--axis ifeval`）

採 IFEval 的作法——只用**程式可判分**的指令（字數上限、分隔符、禁用詞、
JSON 格式、語言限定、恰好 N 句），不需要 LLM judge。自建 17 條涵蓋 zh-TW / ja / en。

自建而非直接用現成資料集的理由：
- **M-IFEval** 只有 fr / ja / es，沒有中文
- **Marco-Bench-MIF** 有 zh 但是簡體，且是 IFEval 的翻譯擴充，繁中在地化不確定
- 本專案只在乎三語，17 條規則型指令就能把「退化成翻譯機」照出來，不值得引入外部相依

### 軸 C：通用能力保留（`--axis general`）

12 題唯一答案問答（三語各 4 題），字串比對判分。額外輸出 `mere_translation_pct`：
自動偵測「模型把問題翻譯掉而不是回答」——判據是答案與問題長度相近、
主要文字系統發生切換、或答案結尾仍是問句。這正是 v2/v3 的招牌失敗模式，
把它變成一個可追蹤的數字。

### 用法

```powershell
uv run python scripts/eval_capability.py --tag base
uv run python scripts/eval_capability.py --tag v3 --adapter outputs/sft-v3
uv run python scripts/eval_capability.py --tag v4 --adapter outputs/sft-v4 --docs 30
```

結果寫 `results/capability/<tag>.json`，逐題明細在 `results/capability/<tag>/`。
單軸執行不會覆蓋其他軸的既有結果（會 merge 進同一份 JSON）。

**耗時**（3070 Ti，linear attention 走 torch fallback）：軸 B + 軸 C 各約 1~2 分鐘，
軸 A `--docs 12` 約 70 分鐘/模型（六方向 × 長文 × max_new_tokens=2048）。
快速迴歸只跑 `--axis ifeval` 與 `--axis general` 即可——這兩軸就足以抓到災難性遺忘。

## 6. 附帶修掉的量測 bug：簡體洩漏率一直被灌水

實作 replay 清洗時發現，專案沿用的洩漏判據 `OpenCC("s2t").convert(s) != s` 是錯的。
`s2t` 目標是「傳統中文」而非「台灣標準」，會把 `剛才→剛纔`、`人群→人羣`、`稽核→稽覈`
這些**正確的台灣用字**轉成傳統異體字，於是判成有簡體。正確判據是 `s2tw`。

已修 `evaluate.py` / `prepare_data.py` / `eval_capability.py` / `build_replay.py`，
並寫 `scripts/rescore_leak.py` 用 `results/hyp/` 的既有譯文重算，**60 筆歷史分數已更正**：

| tag | 方向 | 舊 (s2t) | 新 (s2tw) |
|---|---|---|---|
| **base2bs2tw**-flores | en→zhtw | 5.40% | **0.20%** |
| **base2bs2tw**-flores | ja→zhtw | 4.00% | **0.00%** |
| v3-flores | en→zhtw | 5.20% | 3.60% |
| v3-flores | ja→zhtw | 6.00% | 3.80% |
| v2-flores | en→zhtw | 6.40% | 4.40% |
| base-flores | ja→zhtw | 45.60% | 45.80% |

**出貨組合「官方 2B + s2twp」其實幾乎零洩漏（0.0~0.2%），不是原記的 5.4%**——
CONTEXT 裡「洩漏 5.4%（過閘）」這個數字要改。base 的高洩漏率（45%+）沒什麼變化，
所以「微調大幅修好洩漏」的結論方向不變，只是各版絕對值普遍被灌水 1~3pp。

## 7. 出貨判斷

`CONTEXT.md` 原有結論「官方 Qwen3.5-2B + s2twp 後處理、零訓練，COMET 88.21」
在本次發現後更站得住：微調版不只單句 COMET 沒贏 2B，還把 base 的通用能力與
長篇能力破壞掉。若仍要做 0.8B 微調版，v4 必須先在新面板上證明
**軸 B / 軸 C 不低於 base**，再談翻譯分數。

## 參考

- [Tower: An Open Multilingual LLM for Translation-Related Tasks](https://arxiv.org/abs/2402.17733)
- [Tower+: Bridging Generality and Translation Specialization](https://arxiv.org/abs/2506.17080) — SFT 22% 翻譯 / 78% 通用
- [WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects](https://arxiv.org/abs/2502.12404)
- [M-IFEval: Multilingual Instruction-Following Evaluation](https://arxiv.org/abs/2502.04688)
- [Marco-Bench-MIF: On Multilingual Instruction-Following Capability of LLMs](https://arxiv.org/abs/2507.11882)
- [An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning](https://arxiv.org/abs/2308.08747)
- [GeRe: Efficient Anti-Forgetting in Continual Learning of LLM via General Samples Replay](https://arxiv.org/pdf/2508.04676)
- [Multilingual Contextualization of LLMs for Document-Level MT](https://arxiv.org/abs/2504.12140)
