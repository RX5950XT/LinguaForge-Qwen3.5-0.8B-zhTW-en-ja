# LinguaForge 研究報告 — Qwen3.5-0.8B 繁中／英／日翻譯特化微調

> 從 v1 到出貨版 v5e 的完整紀錄：每一版改了什麼、量到什麼、為什麼往下一版走。
> 逐項發現的原始紀錄（含被推翻的中途假設）在 `RESEARCH-v5.md`；踩坑通則在 `tasks/lessons.md`。
> **最終數字以 `results/` 的 json 為準**——本文所有表格都可回溯到檔案，沒有只活在文件裡的數字。

## 摘要

| | |
|---|---|
| 基座 | `Qwen/Qwen3.5-0.8B`（873M，Apache-2.0，混合線性注意力 + 視覺塔） |
| 方法 | LoRA SFT（r=64/α=128），六方向 zh-TW ↔ en ↔ ja |
| 資料 | 502,993 筆翻譯（公開語料，含文件級）＋ 35,177 筆通用指令 replay |
| 硬體 | 單張 RTX 5060 Ti 16GB（v1~v3 是 3070 Ti 8GB），10.5 小時、peak VRAM 4.00GB |
| 出貨版 | **v5e**，六條硬閘機器判定全過（`regression_guard.py` exit 0） |
| 成果 | FLORES 全量 COMET **84.64 → 86.56**；簡體洩漏 en→zhtw **10.18% → 1.09%**、ja→zhtw **43.58% → 0.69%** |

三個貫穿全程的結論：

1. **0.8B 的病灶是字形與用語，不是語意。** 零樣本 COMET 已有 84.64，但輸出中文時
   一成到四成的行摻簡體字。SFT 修的是這個。
2. **每一次「進步」都要先確認對照組跟候選同條件。** 本專案兩度因為對照組沒對齊
   （n=500 vs 1012、有無 `no_repeat_ngram`）而誤判勝負，兩次都自己抓回來。
3. **翻譯特化最貴的代價是通用能力，而且預設的評測面板看不到它。** 五個翻譯基準
   全綠的 v3，實際上已經變成「翻譯函數」——問它問題會把問題翻譯掉。

---

## 1. 問題定義與零樣本地板

需求：本機、低成本、純開源語料，做繁體中文（臺灣用語）↔ 英文 ↔ 日文六方向翻譯。

先量官方模型零樣本（這一步後來成為本專案最重要的方法論教訓，見 §4.3）：

| | COMET | 簡體洩漏 |
|---|---|---|
| 語意 | 82~86，**堪用** | — |
| 字形 | — | en→zhtw 10~21%、ja→zhtw **43.6~45.6%** |
| 日文互譯 | chrF++ 明顯偏低 | — |

→ 策略定為「修字形 + 補日文方向」，不是硬拚語意天花板。

## 2. 版本演進總表

⚠️ **不同區塊之間不可直接比較**：樣本數與解碼設定變過兩次。同區塊內才是 apples-to-apples。

### 2.1 v1 → v3（n=500、greedy、3070 Ti 8GB）

| 版本 | 關鍵改動 | FLORES COMET | 洩漏 en→/ja→zhtw |
|---|---|---|---|
| base | 官方零樣本 | 83.85 | 20.8% / 45.6% |
| v1 | r32/α64、2 epoch、en↔ja 用 OPUS-100 | 83.06 | 8.2% / 5.8% |
| v2 | en↔ja 換乾淨書面語料 | 84.58 | 6.4% / 6.4% |
| **v3** | r64/α128 + NEFTune、多領域 + 污染閘、773k 筆 | **84.80** | 5.2% / 6.0% |

- **v1 的教訓**：洩漏修好但 COMET 掉 0.75，日文方向退步。我一度歸因為「翻譯特化的固有
  trade-off」——**錯的**。真因是 en↔ja 全靠 OPUS-100（口語、雜訊、對齊差），而 FLORES 測
  正式書面日文，domain 不匹配。換乾淨語料（WikiMatrix / JParaCrawl / KFTT / Tatoeba /
  News-Commentary）重訓即反超基線。
- **v3 的多領域診斷**：擴到五個基準（FLORES / NTREX / WMT22 / ALT / TICO-19）後才看得出
  v2 對新聞語域有過擬合（NTREX 一致低 1.3~5.8）、WMT into-English 崩（源側解析弱）、
  而醫療與 Wikinews 反而穩。**只有一張考卷時，任何提升都無法否證是不是過擬合到它。**
- **2B 對照**：並行訓了 2B QLoRA（NF4，8GB 唯一塞得下的方式）。補測官方 2B 零樣本後翻轉：
  官方 2B 每個基準 COMET 都高於我們的 2B 微調版，而洩漏可用 OpenCC `s2twp` 後處理零成本修掉。
  → **2B 這條線的最佳解是「官方模型 + 後處理」，不是微調。**
  但後續 NF4 2×2 對照證明那 0.65 的差距大半是 4-bit 量化稅（−0.69），同精度下微調 ≈ 中性（−0.06），
  **不是微調破壞品質**。真正判定需雲端全精度 bf16 重訓 2B（未做，超出本機範圍）。
- **量化稅（0.8B）**：微調與量化正交可疊加。同精度微調增益恆為 ~+1.0，且**台灣正體字形
  在 4-bit 下幾乎零損失**（離散 token 偏好不靠權重精度）→ v3-nf4 的字形正確性完勝官方 bf16。

### 2.2 v4：五個基準全綠，模型卻壞了

使用者手測 v3 覺得「怪怪的」，實測確認**災難性遺忘**：

| 缺陷 | 實際表現 |
|---|---|
| 通用能力歸零 | 「台灣最高山是哪一座？」→ 把問題翻成繁中丟回來，不回答 |
| 長篇腰斬 | 5 段 1491 字的文章只翻前 2 段就發 `<im_end>` |
| 無限重複 | v2 的 zhtw→ja 對長文輸出同一句 30+ 次 |

根因是配比，不是 rank 也不是 epoch（v3 r64/1ep 與 v2 r32/2ep 崩得一樣）：
**訓練集 100% 翻譯任務、0% 通用指令；100% 單句對、0% 多段落。**
對照 SOTA，Tower+（Unbabel 2025）的 SFT 是 22% 翻譯 / 78% 通用指令；文獻另載
只 replay 1% 歷史指令資料就足以守住指令跟隨。我們用了 0%。

v4 的修法：
- **通用 replay**（`scripts/build_replay.py`）：oasst2 / aya_dataset / oasst2-33k-ja，
  皆 Apache-2.0，去重後 35,177 筆。**繁中只有 1,453 筆**——這個缺口至今沒補上（見 §5）。
  `aya_collection` 的 traditional_chinese 有 360 萬筆但是英文 Flan 的機翻，
  連編號都被當文字翻譯（`1.` → 「一,他們」），混進去會毒化輸出格式，已排除。
- **文件級樣本**：既有語料本來就有文件邊界，按邊界重拼成多段落樣本，不需要新語料。
- **關掉 packing**：TRL 的 bfd packing 會自動開 padding-free，而那只支援 FlashAttention；
  Windows 用 sdpa 等於把多筆樣本接成一條長序列（實測改前一個樣本、後一個樣本 logits 差 6.6250）。
  Qwen3.5 的線性注意力遞迴狀態更是切不斷，裝了 flash-attn 也修不掉。
- **token 預算組批**：關掉 packing 後固定 batch size 只能遷就最長樣本，micro-batch 平均只裝
  250 token（上限 1450）。改成依 token 預算動態組批後 1.75 → 9.4 samples/s，訓練 48h → 7h。
- **三軸能力面板**（`scripts/eval_capability.py`）：文件級完整度 / 可驗證指令跟隨 / 通用能力保留，
  補上翻譯基準測不到的維度。

### 2.3 v5a → v5c：資料品質三連（n=1012、greedy）

| 版本 | 唯一變因 | COMET |
|---|---|---|
| v4 | （對照起點） | 84.70 |
| v5a | 注水式來源配額 + `DOC_MAX` 6→16 + `max_length` 1408 | 84.85 |
| v5b | LaBSE 雙語語意過濾 ≥ 0.60 | 85.09（epoch 對齊值） |
| v5c | LaBSE 門檻 0.65 | 85.26 |

- **v5a 修的是配方截斷**：`--limit 20000` × `MAX_SHARE 0.5` 讓依序貪婪的來源迴圈只餵得到
  前 2 個語料，每方向的領域組合整個消失。改成注水式配額後領域數 2~3 → 5~7。
  只買到 en→zhtw +1.36，其餘五向持平 → 假設只成立一半。
- **v5b/v5c 修的是語意錯位**：規則式清洗抓不到「兩句根本沒關係」。LaBSE 掃全 20 份語料，
  `globalvoices.ja-zhtw` 有 64.7% 的行相似度低於 0.60（對照 `jparacrawl` 0.8%）。
  **按實際配額加權後，錯位率與四個 en-pivot 方向的成績完全單調對應。**
  門檻用 FLORES 黃金對齊校準（三語言對 p01 都 ≥ 0.641，0.60 砍不到正確樣本）。
  合計只換到 +0.41，方向對但力道不夠——真病灶在別處。
- **副產物教訓**：`load_best_model_at_end` 依 eval_loss 挑 checkpoint 挑錯了（差 0.0003 的雜訊，
  實測 COMET 低 0.29）→ v5c 起關閉。

### 2.4 解碼端：零訓練成本 +0.92

比 v5c 整整 7h49m 的訓練所得（+0.17）高一個量級：

| 設定 | 作用 |
|---|---|
| `num_beams=4` | 修漏譯（F8 量到缺口全在長度尾巴上） |
| `no_repeat_ngram_size=4` | 修 beam 帶來的重複；**英文不用**（合法 4-gram 重複多） |
| zh-TW 目標**不可**用 `repetition_penalty` | 它重新加權單字時會動到繁簡異體字選擇，實測洩漏 4.65% → 13.06% |

`evaluate.DECODE` 是這組設定的唯一真相來源，評測與能力面板都 import 它，結果 json 記 `decode_defaults`。

### 2.5 v5d → v5f：資料量、然後是容量（n=1012、beam4 + 逐語言 nrng）

| 版本 | 設定 | 筆數 | eval_loss | COMET | 通用能力 |
|---|---|---|---|---|---|
| base | 官方零樣本 | — | — | 84.64 | 基準 |
| v5d | r64 / 1 epoch | 272,783 | 1.831 | 86.34 | **兩軸最佳** |
| **v5e** | r64 / 1 epoch | **502,993** | **1.7708** | **86.56** | 兩軸 ≥ base |
| v5f | **r128** / 1 epoch | 502,993 | 1.7440 | 86.73 | **BELEBELE 崩，不得出貨** |

- **v5d 證實了資料量假設**：v5c 的第二個 epoch 是在背資料（train loss 1.856→1.635，
  eval_loss 卡在 1.877→1.866）。一個 epoch 的全新樣本贏過同一批看兩次，且只花半個算力。
- **v5e 加到 80k/方向仍有效但開始收斂**：×1.84 只換到 +0.23，且 ja↔zhtw 首次抽不滿
  （75,105 / 80,000，文件級只湊到 7,105）。**純加量這條路到此為止。**
- **v5f 是本專案唯一被硬閘攔下的版本**，也是引入外部基準的價值所在：

  | BELEBELE（輪轉去偏，各 900 題） | base | v5d | v5e | v5f |
  |---|---|---|---|---|
  | zh-TW | 55.81 | 57.25 | 57.28 | **51.28** ❌ |
  | ja | 51.78 | 50.31 | 52.36 | **43.94** ❌ |
  | en | 57.83 | 68.08 | 62.81 | 65.19 |

  機制看得到：v5f 在 zh-TW / ja 押同一個選項 55.2% / 56.9%（英文 28.6%，接近均勻）
  ——**中日文已經退回固定先驗，不太在作答**。門檻是 base − 3.0，兩格分別超出 4.53 / 7.84。
  **r=64 的兩版兩軸都 ≥ base，資料量 ×1.84 沒有代價；只有 r=128 崩 → 退化的變因是 r，
  是斷崖不是斜坡。** 自建的 n=90 面板完全沒抓到（還顯示上升），因為題目形式離訓練資料太近。

---

## 3. 出貨版 v5e 的完整成績

### 3.1 翻譯（FLORES-200 devtest 全量 n=1012，beam=4 + 逐目標語言解碼）

對照組是**同樣本數、同解碼設定**的官方 `Qwen/Qwen3.5-0.8B`（`results/baseline/base-full-flores.json`）。

| 方向 | chrF++ (base→v5e) | BLEU (base→v5e) | COMET (base→v5e) | 簡體洩漏 (base→v5e) |
|---|---|---|---|---|
| en→zhtw | 19.38 → 20.26 | 25.57 → 28.90 | 86.32 → 86.23 | **10.18% → 1.09%** |
| zhtw→en | 47.78 → 50.33 | 19.34 → 23.72 | 84.79 → **85.27** | — |
| en→ja | 20.26 → 24.15 | 19.81 → 22.65 | 86.17 → **88.44** | — |
| ja→en | 45.67 → 50.17 | 16.67 → 22.70 | 84.85 → **86.20** | — |
| ja→zhtw | 10.53 → 16.60 | 10.38 → 23.07 | 81.44 → **86.10** | **43.58% → 0.69%** |
| zhtw→ja | 14.80 → 18.48 | 12.97 → 16.93 | 84.24 → **87.14** | — |
| **均值** | 26.40 → **30.00** | 17.46 → **22.60** | 84.64 → **86.56** | |

> **en→zhtw 是持平，不是輸。** paired bootstrap（n=1012、1000 次配對重抽）Δ = −0.086，
> 95% CI [−0.414, +0.225]，跨 0 → 與官方模型量不出差別。
> 另：官方那 86.32 是在 10.18% 簡體洩漏下拿到的（COMET 對簡繁不敏感），兩邊產出的不是同一種東西。

### 3.2 通用能力（外部公開基準）

| 基準 | base | v5e |
|---|---|---|
| BELEBELE zh-TW / ja / en（各 900 題，輪轉去偏） | 55.81 / 51.78 / 57.83 | **57.28 / 52.36 / 62.81** |
| 知識 TMMLU+ / MMMLU-JA / MMLU（各 900 題） | 30.67 / 36.08 / 43.67 | 30.53 / **36.64** / 43.81 |
| 自建通用問答 n=90 | 78.9 | 72.2 ⚠ |
| 指令遵循 ifeval n=90 | 53.3 | 48.9 ⚠ |

**六格外部基準全部 ≥ base**（BELEBELE 三語甚至皆升）。落後的是兩個自建面板，同源於 replay 不足。

> 選擇題一律**輪轉選項 ×4**：0.8B 對答案字母有強先驗（base 押同一字母 43%、v5f 63%，隨機應為 ~27%），
> 單輪計分量到的是先驗不是知識。比選項文字 logprob 那條路實測貼著隨機基準，已排除。
> 汙染立場：沒有任何公開基準能證明未被 Qwen3.5 預訓練汙染，但本專案量的是**同一份題目上
> base → finetune 的差值**，汙染兩邊共有、相減抵消 → 絕對分數不得對外宣稱能力，只能用 Δ。

### 3.3 長文件（25 篇 × 6 方向）

| 指標 | base | v5e |
|---|---|---|
| 行數對齊比（中位） | 1.375 ~ 1.500 | **1.000（六向皆是）** |
| 尾段譯出比（中位） | 1.101 ~ 1.646 | 0.851 ~ 0.924 |
| 腰斬率 | 0 ~ 4% | 0 ~ 4% |

**v2/v3 的「長文只翻前兩段」在 v5e 已消失**（v3 是 completeness 0.103、腰斬率 91.7%）。
意外的是反面：**base 才是尾段超譯**（翻完不停繼續生成），微調把這個老毛病修掉了。

### 3.4 出貨路徑（GGUF）跑的不是被評測的解碼設定

`llama-cli` 沒有 beam search 也沒有 `no_repeat_ngram`，只能 greedy。實測 en→zhtw 前 100 句：

| 系統 | chrF++ | BLEU | 簡體洩漏 | 與 f16 完全相同 |
|---|---|---|---|---|
| GGUF f16（greedy） | 20.50 | 27.83 | 2.00% | 100% |
| GGUF Q8_0 | 20.21 | 27.28 | 2.00% | 63% |
| GGUF Q4_K_M | 19.49 | 26.07 | 2.00% | 12% |
| bf16 + beam=4（評測路徑） | 20.00 | 28.48 | **0.00%** | 5% |

- 量化稅單調（Q8_0 −0.29、Q4_K_M −1.01 chrF++），**greedy 沒有造成漏譯**（長度比與 beam=4 同帶）。
- 真正的代價是**簡體洩漏 0% → 2%**：greedy 逐 token 挑字會在同一行內混用字形
  （`美國國家体操隊`），beam=4 比整條路徑總分才壓得住 → **GGUF 路徑務必補 `s2twp` 後處理**
  （實測 2.00% → 0.00%，chrF++ 只掉 0.04）。

### 3.5 F57 長口語刷屏與 decode_search（2026-08）

**F57**：長口語／社群文 →en 句級無限重複。根因不是「只有 context 不夠」，而是舊
`DECODE` 對 **en 未開** `no_repeat_ngram_size`（ja/zhtw 已是 4）。消融：
單靠分段不夠；en 開 nrng=3/4/6 全止血。已落地 `DECODE["en"]` 與 ja 對齊
（`repetition_penalty=1.1` + `no_repeat_ngram_size=4`）。**GGUF／裸 greedy 仍會炸**，
勿當對等路徑。

**decode_search**（`scripts/decode_search.py`，`results/decode_search/`）：在 **只動
v5e 解碼、不重訓** 下 grid 10 候選 × FLORES n=200，硬閘含 long-oral loop、zhtw 洩漏、
方向 chrF 不得大崩。winner **`b4_lp1.2`**：

| 項 | 出貨 |
|---|---|
| beams | 4 |
| `length_penalty` | **1.2**（原 1.0） |
| DECODE | 三語 nrng=4；en/ja + rep 1.1 |

無 nrng 的高 chrF 被 long-oral 打掉；`nrng=6` 被洩漏閘打掉。n=500 確認
lp1.2 vs 1.0 的 chrF++ AVG **Δ≈+0.12**。應用接入清單見 [`INTEGRATION.md`](INTEGRATION.md)。

## 4. 方法論：本專案自己抓到的量測錯誤

寫在這裡是因為每一條都曾經產出過「看起來很合理但錯的結論」。

1. **對照組沒跟候選同條件，等於沒有對照組。**「六方向 COMET ≥ base」這道閘從 v5c 用到 v5f，
   比的是 n=500 無 nrng 的 base 對 n=1012 有 nrng 的候選。照候選條件重跑 base 後
   en→zhtw 從 86.19 變 86.32，「v5d 首度贏過 base」直接翻轉成落後 0.09。
   更糟的是那個 86.19 **沒有機器可讀的來源**——`base-b4-flores.json` 的六個 comet 欄位全是 `null`，
   數字只活在研究文件的表格裡。
2. **沒有 CI 的點估計不能宣告勝負。** 修好對照組後的 −0.09，CI 是 [−0.414, +0.225]，跨 0。
   也就是說「+0.04 領先」與「−0.09 落後」兩個都在雜訊裡——我用一個假訊號推翻了另一個假訊號。
   → 硬閘改成三段判定，落在雜訊帶要跑 `paired_bootstrap.py`，且 CI 必須落檔。
3. **簡體洩漏指標本身壞過兩次。** 先是用 `s2t`（目標是「傳統中文」不是「台灣標準」，
   把「剛才／人群／稽核」判成洩漏），改 `s2tw` 後修正 60 筆歷史分數；後又發現 round-trip
   量到的是異體字偏好，最終改成簡體專用字集 − 台灣正字白名單。
4. **一個混合多種病的指標會讓病互相抵消。** `completeness_median`（整篇字元比）同時受
   「行文精簡」「多吐垃圾行」「尾段腰斬」影響，base 靠第二項補償第三項拿到漂亮分數。
   拆成 `line_ratio` 與 `tail_ratio` 後真實行為才顯現。**訂硬閘前先問：這個數字變差，
   可能的原因有幾種？超過一種就不能拿來當閘。**
5. **「人工核對」不是閘。** 規範文件寫了六條硬閘，守門腳本只實作兩條半，而漏掉的兩個 COMET
   方向裡正好包含唯一有爭議的 en→zhtw。→ 六條全部機器化，**缺值用獨立 exit code 跟 PASS 分開**。
6. **微調前先量官方零樣本地板。** 否則會把「大模型本來就強」記成「我們微調的功勞」——
   2B 那條線就是這樣白燒了 17 小時 GPU 才發現微調是淨負值。

## 5. 收工判斷：為什麼不再訓練下一版

五個可動的變因逐一結案：

| 槓桿 | 狀態 | 依據 |
|---|---|---|
| 資料量 | 走完 | ×1.77 → +0.25、×1.84 → +0.23，且 ja↔zhtw 已抽不滿 |
| LoRA r | 上界 r ≤ 64 | r=128 BELEBELE 中日文 −4.53 / −7.84，斷崖 |
| replay 擴充 | 卡授權 | 三個來源抽乾，找不到 Apache-2.0 相容、非機翻的 zh-TW 指令集 |
| 解碼 | 已收割 | +0.92 COMET，零訓練成本 |
| epoch | 已證偽 | v5c 兩 epoch/141k（1.866）輸 v5d 一 epoch/273k（1.831） |

唯一沒探過的是 r=96。**就算樂觀假設它線性內插，也只值 +0.08~0.12 COMET（→ 86.64~86.68），
離目標 87.00 還差 0.32 以上**；而 r 是斷崖不是斜坡，連內插這個前提都不成立，
BELEBELE 的下檔無法預測。11 小時 GPU 換一個碰不到門檻、下檔不可測的點——不跑。

**要再往上必須先有新語料，不是先有新訓練。** 兩個缺口都指向同一堵牆：
en→zhtw 需要新的 zh-TW **書面語**來源（現有 zh-TW 目標側 100 萬筆主要來自 COCT 與字幕），
通用能力需要授權相容的 zh-TW **指令**語料。兩者都找不到。

## 6. 復現

```powershell
uv sync                                      # 主環境；COMET 另裝：uv sync --project tools/comet
uv run python scripts/download_data.py       # 語料（冪等，可中斷重跑）
uv run python scripts/prepare_data.py --limit 80000 --dev-from data/sft/dev.jsonl
uv run python scripts/train_sft.py --config configs/sft_lora_v5e.yaml
uv run python scripts/evaluate.py --tag v5e --adapter outputs/sft-v5e --full
uv run --project tools/comet python tools/comet/score.py --tag v5e-flores
uv run python scripts/eval_bench.py      --tag v5e --adapter outputs/sft-v5e
uv run python scripts/eval_capability.py --tag v5e --adapter outputs/sft-v5e --axis all
uv run python scripts/regression_guard.py --candidate v5e          # exit 0 才可出貨
```

![v5e SFT loss](assets/loss_curve.png)

語料一律用上列腳本重建（授權各異，不 re-host）。大檔（權重、GGUF）發布於 Hugging Face。

## 參考

- [Tower+: Bridging Generality and Translation Specialization](https://arxiv.org/abs/2506.17080) — SFT 22% 翻譯 / 78% 通用
- [Tower: An Open Multilingual LLM for Translation-Related Tasks](https://arxiv.org/abs/2402.17733)
- [An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning](https://arxiv.org/abs/2308.08747)
- [GeRe: Efficient Anti-Forgetting in Continual Learning of LLM via General Samples Replay](https://arxiv.org/pdf/2508.04676)
- [Multilingual Contextualization of LLMs for Document-Level MT](https://arxiv.org/abs/2504.12140)
- [WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects](https://arxiv.org/abs/2502.12404)
- [BELEBELE: a parallel reading comprehension dataset in 122 language variants](https://arxiv.org/abs/2308.16884)
