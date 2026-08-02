# LinguaForge v5e — 接入應用指南與修復提示詞

出貨版：`outputs/sft-v5e` / HF `RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja`  
解碼真相來源：`scripts/evaluate.py` 的 `DECODE` / `NUM_BEAMS` / `LENGTH_PENALTY`  
證據：`results/decode_search/`（2026-08-02）、F57 長口語止血

---

## 1. 出貨解碼（必對齊）

| 參數 | 值 | 原因 |
|---|---|---|
| `num_beams` | **4** | 修漏譯；greedy 品質較差 |
| `length_penalty` | **1.2** | decode_search winner（n=200 過閘，n=500 ΔchrF++≈+0.12） |
| `do_sample` | **false** | 翻譯用確定性解碼 |
| `eos_token_id` | **`[248046, 248044]`**（`<\|im_end\|>` + `<\|endoftext\|>`） | SFT 用 im_end 收尾；只設 endoftext 會灌水 |
| 目標 **ja** | `repetition_penalty=1.1`, `no_repeat_ngram_size=4` | 抑重複 |
| 目標 **en** | 同上 | **F57**：長口語→en 無 nrng 會句級刷屏 |
| 目標 **zhtw** | **僅** `no_repeat_ngram_size=4` | **禁止**對 zhtw 開 `repetition_penalty`（洩漏會暴增） |

提示詞格式（與訓練一致）：

```
system: You are a professional translator.
user:   翻譯成繁體中文：\n{原文}   # 或 翻譯成英文： / 翻譯成日文：
```

載入：`AutoModelForImageTextToText` + chat template；勿當純 CausalLM 亂拼 prompt。

### ⚠ generation prompt 必須以空 think 區塊收尾

模型送進去的完整字串**一定**要長這樣，最後 4 個 token 不可少：

```
<|im_start|>system\nYou are a professional translator.<|im_end|>\n
<|im_start|>user\n翻譯成繁體中文：\n{原文}<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n        ← 248068, 271, 248069, 271
```

`chat_template.jinja` 在 `enable_thinking` 未開時走 else 分支固定補
`<think>\n\n</think>\n\n`，訓練與評測全程如此。transformers 的
`apply_chat_template(..., add_generation_prompt=True)` 會自動補；
**llama.cpp 系不會**——node-llama-cpp 的內建 `QwenChatWrapper`（`thoughts` 六個選項都試過）
以及未加 `--jinja` 的 `llama-cli` 都只輸出到 `<|im_start|>assistant\n` 為止。

少這 4 個 token 的實測後果（f16 全精度，同 greedy，30 句樣本）：

| 指標 | 缺 think | 補 think |
|---|---|---|
| 憑空標籤前綴（`說明：`／`問：`／`1. `／`選擇：`／`故事說，`） | 9 句 | **0** |
| 拉丁專名保留率（Q8_0） | 73.3% | **93.3%** |
| 原文無年份卻生出年份 | 2 句 | **0** |
| 缺陷總數（Q4_K_M） | 20 | **5** |

例：`The NVIDIA H200 has 141GB of HBM3e memory.`
→ 缺：「141GB HBM3e 記憶體。」／補：「NVIDIA H200 擁有 141GB HBM3e 記憶體。」

修法：llama-cli 加 `--jinja`；node-llama-cpp 用自訂 wrapper——

```js
class Qwen35ChatWrapper extends QwenChatWrapper {
  generateContextState(options) {
    const state = super.generateContextState(options)
    const last = options.chatHistory[options.chatHistory.length - 1]
    if (last?.type === 'model' && (last.response == null || last.response.length === 0))
      state.contextText = LlamaText([state.contextText, '<think>\n\n</think>\n\n'])
    return state
  }
}
```

`--reasoning off --reasoning-budget 0` 解的是「不要解析 thinking 輸出」，
**補不了這 4 個 token**，兩者要一起做。完整證據見 `docs/DEFECT-AUDIT-2026-08-03.md`。

---

## 2. 已知怪行為（設定 vs 本質）

### 設定用錯就會出現

| 症狀 | 常見根因 | 修法 |
|---|---|---|
| 譯完後無限灌水到 max tokens | 只設單一 eos | 雙 EOS 如上 |
| 長口語／社群文→en（或 greedy→ja）同一句刷屏 | 未開 `no_repeat_ngram_size`；或 GGUF greedy | transformers 走 DECODE；GGUF 加 `--repeat-penalty` + 分段 |
| 譯文前綴出現 `A. `、思考句、日文雜訊 | thinking 沒關 | llama-cli：`--reasoning off --reasoning-budget 0`（舊 kwargs 會被靜默忽略） |
| 譯文憑空多出 `說明：`／`問：`／`1. `／`選擇：`／`圖為`／`故事說，` | **generation prompt 少了 `<think>\n\n</think>\n\n`** | 見 §1「⚠ generation prompt 必須以空 think 區塊收尾」；**不要靠後處理剝前綴** |
| 專名整個消失（`NVIDIA H200`→「141GB 記憶體」）、原文無年份卻生年份 | 同上 | 同上 |
| 繁中夾簡體字（GGUF 更明顯） | greedy 無 beam；未後處理 | 正式路徑用 beam=4；GGUF 必 `OpenCC("s2twp")` |
| 開了 rep-penalty 後繁中更亂 | 對 **zhtw** 用了 `repetition_penalty` | 只對 en/ja 開 |
| Windows 提示詞變亂碼 | llama-cli `-p` + cp950 | 改 `-f prompt.txt`（UTF-8） |
| 長文尾段消失／不完整 | 超 context 或 max_new_tokens 太小 | 分段翻譯（建議源段 ≤250–300 字元級再拼）；調高 max_new_tokens |

### 正確設定下仍有的限制

- **en→zhtw** 語意與 base 統計持平（字形／正體明顯較好）
- 0.8B：罕見專名、強世界知識句仍可能平庸或胡謅
- **不是通用助手**（問答／指令遵循會略弱於 base）
- GGUF **無法對等** PEFT+beam 路徑（無 beam、無 nrng）

---

## 3. 給另一個 AI／工程師的「修復翻譯功能」提示詞

以下整段可直接貼到負責你的翻譯應用的 agent／工程師：

```
你要修復一個接入 LinguaForge-Qwen3.5-0.8B-zhTW-en-ja（v5e LoRA）的翻譯功能。
模型是「翻譯特化」，不是聊天助手。優先對齊出貨解碼，不要發明新 prompt 風格。

## 權威來源
- 解碼常數：訓練倉庫 scripts/evaluate.py → DECODE / NUM_BEAMS=4 / LENGTH_PENALTY=1.2
- 模型卡：HF RX5950XT/LinguaForge-Qwen3.5-0.8B-zhTW-en-ja 的 README
- 接入說明：訓練倉庫 docs/INTEGRATION.md
- base：Qwen/Qwen3.5-0.8B；載入類別必須是 AutoModelForImageTextToText（transformers≥5.x）

## 必改清單（缺一可能「怪怪的」）

0) generation prompt 必須以空 think 區塊收尾（**最重要，先做這個**）
   送進模型的完整字串一定要長這樣，最後 4 個 token 不可少：

     <|im_start|>system\nYou are a professional translator.<|im_end|>\n
     <|im_start|>user\n翻譯成繁體中文：\n{原文}<|im_end|>\n
     <|im_start|>assistant\n<think>\n\n</think>\n\n

   token id 是 248068, 271, 248069, 271（<think> / \n\n / </think> / \n\n）。
   chat_template.jinja 在 enable_thinking 未開時走 else 分支固定補這段，
   模型從頭到尾是帶著它訓練與評測的。

   - transformers：apply_chat_template(..., add_generation_prompt=True) 會自動補，不必動。
   - llama.cpp / node-llama-cpp / Ollama：**不會補**。
     node-llama-cpp 內建的 QwenChatWrapper 是照 Qwen3 寫的，Qwen3.5 的 else 分支不同，
     thoughts 的六個選項（auto/discouraged/disabled/none/forced/open）沒有一個會補。
     llama-cli 要加 --jinja 才會吃 GGUF 內建 template。

   少這 4 個 token 的實測後果（f16 全精度、同 greedy、30 句樣本）：
     憑空標籤前綴（說明：／問：／1. ／選擇：／故事說，）9 句 → 0
     拉丁專名保留率（Q8_0）73.3% → 93.3%
     原文無年份卻生出年份 2 句 → 0
     缺陷總數（Q4_K_M）20 → 5
   例：The NVIDIA H200 has 141GB of HBM3e memory.
       缺 → 「141GB HBM3e 記憶體。」（NVIDIA、H200 全消失）
       補 → 「NVIDIA H200 擁有 141GB HBM3e 記憶體。」

   node-llama-cpp 的修法（不是後處理）：

     class Qwen35ChatWrapper extends QwenChatWrapper {
       generateContextState(options) {
         const state = super.generateContextState(options)
         const last = options.chatHistory[options.chatHistory.length - 1]
         if (last?.type === 'model' && (last.response == null || last.response.length === 0))
           state.contextText = LlamaText([state.contextText, '<think>\n\n</think>\n\n'])
         return state
       }
     }
     new LlamaChatSession({ contextSequence, chatWrapper: new Qwen35ChatWrapper() })

   注意：--reasoning off / --reasoning-budget 0 / budgets.thoughtTokens=0 解的是
   「不要解析或生成 thinking 輸出」，**補不了這 4 個 token**，兩件事都要做。

   改完先驗證 prompt 本身：把 wrapper 產出的字串印出來，
   跟 transformers apply_chat_template 的輸出逐字元比對，一致才往下修別的。

1) 雙 EOS
   eos_token_id = [im_end_id, endoftext_id]  # 通常 [248046, 248044]
   只用 config 預設 eos 會在正確譯文後繼續生成到 max_new_tokens。

2) 出貨 generate 參數
   num_beams=4, length_penalty=1.2, do_sample=False
   依「目標語」套用：
   - tgt en 或 ja: repetition_penalty=1.1, no_repeat_ngram_size=4
   - tgt zhtw / zh-TW: 只有 no_repeat_ngram_size=4
     （禁止對繁中目標加 repetition_penalty，會攪亂繁簡選字、洩漏飆升）

3) Chat 格式（與 SFT 一致）
   system: "You are a professional translator."
   user: f"{指令}\n{原文}"
   指令僅允許：
   - 翻譯成繁體中文：
   - 翻譯成英文：
   - 翻譯成日文：
   使用 tokenizer.apply_chat_template(..., add_generation_prompt=True)
   decode 時 skip_special_tokens=True，且只取 generated 段（去掉 prompt）。

4) 長文
   - 不要整篇一次硬塞到爆 context。
   - 按段落／句群分段（建議單段源文不要過長，約 250–300 CJK 字或等價），逐段翻譯再串接。
   - max_new_tokens 依段長設足（至少 ≈ 源 token 的 1.5–2×，設上限防 runaway）。
   - 分段 alone 不能治 en 刷屏；仍須 no_repeat_ngram_size（見上）。

5) 若走 GGUF / llama.cpp / Ollama 等（非 transformers beam）
   - 這不是評測主路徑：無 beam、通常無 no_repeat_ngram。
   - **先確認第 0 點的 think 前綴有補**，這條路徑預設不補，是最大的品質落差來源。
   - 必關 thinking：--reasoning off --reasoning-budget 0（舊 enable_thinking 可能被靜默忽略）。
   - 精度選 Q8_0：Q4_K_M 實測會把罕見專名音譯掉（Kimi→金剛、Sol→索爾），
     f16 與 Q8_0 都保留；這一項才是真正的量化稅，其餘缺陷跟量化無關。
   - 加 --repeat-penalty 1.1（或 runtime 等價）；長文更要分段。
   - 目標為繁中時，輸出後強制 OpenCC s2twp 後處理（greedy 會同句混繁簡）。
   - Windows 不要用會被主控台編碼打壞的 -p 餵 CJK；改檔案餵入 UTF-8。
   - 追求品質請改走 PEFT 或 merged-bf16 + 第 2 點參數。

6) 後處理
   - 去掉首尾空白；可 strip 模型偶發的引號包層。
   - 繁中路徑：OpenCC("s2twp") 作為保險（beam 路徑洩漏已很低，GGUF 必做）。
   - 偵測「同一 n-gram 連刷」：若仍出現，視為解碼參數未生效，修 runtime 不要先怪權重。

7) 不要做的事
   - 不要當 general chat system prompt 大改人設。
   - 不要 temperature>0 當預設翻譯。
   - 不要對 zhtw 開 high repetition_penalty。
   - 不要假設「只有 context 截斷才會壞」——缺 EOS / 缺 nrng 在短中文也會炸。
   - **不要用後處理 regex 剝掉「說明：」「問：」「1. 」這類前綴**。那是止血。
     這些前綴代表 prompt 錯了，模型在跑另一條分布，你剝掉的只是最顯眼的症狀，
     同時發生的專名消失與年份幻覺 regex 抓不到。先修第 0 點再談要不要留後處理。
   - 不要因為「量化三個精度數字都差不多」就判定是權重／語料問題。
     f16 也會發生，只證明不是量化，不證明不是 prompt。

## 驗收（修完必跑）
先跑舊有的五項 smoke：
A. The patient should take this medication twice a day. → 繁中，完整一句、無簡體、無灌水。
B. 週末的夜市人聲鼎沸。 → 英文，一句、無重複尾巴。
C. 長口語／社群風多句中文 → 英文：不得同一子句無限重複；允許分段。
D. 正常句不得貼滿 max_new_tokens（否則 EOS 未生效）。
E. 繁中輸出抽樣無簡體專用字（或 s2twp 後為 0）。

再跑 30 句客觀指標，**修前修後各一次，附實際輸出對照**：
   訓練倉庫  uv run python scripts/bench_defects.py --label <你的標籤>
   應用端    node scripts/verify-chat-wrapper-fix.js [gguf]   （樣本與指標共用 bench-cases.js）

門檻（同一組 30 句、同一組指標）：
   憑空標籤前綴          = 0
   拉丁專名保留率        ≥ 90%
   原文無年份卻生出年份  = 0
   多行輸入行數保留率    ≥ 95%
   缺陷總數              < 8

長度比門檻要依語系分段（英→中正常落在 0.2~0.3），寫死 0.3~3 會把翻得好的句子
判成缺陷——實測 30 句裡 12 句是這樣誤報的，佔缺陷總數一半以上。

已知**修不掉**的兩項，不要為了它們亂改 prompt：
   多行且各行互不相關（規格表／清單）只會譯出第一行 → 訓練裡的多行樣本
     全是連貫段落，這是分布外，要補語料。
   open weight / agentic coding 等 2023 後 AI 術語未學到 → 同上。

## 實作要求
- 先定位應用裡 generate / llama 呼叫點，對照上表 diff，最小改動修到過驗收。
- 把 DECODE 常數集中成一處（依 target lang 查表），避免三處複製不一致。
- 修完用實際 log 證明：打印送進模型的完整 prompt 字串（或 token id）、
  eos_token_id、num_beams、length_penalty、no_repeat_ngram_size、是否 s2twp。
- 每一項改動都要能在同一組 30 句上前後對照，附實際輸出，不接受「感覺變好了」。
```

---

## 4. 最小可跑參考（transformers）

見模型卡 README「快速使用（PEFT）」；核心是：

```python
EOS = [248046, 248044]  # 或從 tokenizer 轉 id
DECODE = {
    "ja": {"repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    "en": {"repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    "zhtw": {"no_repeat_ngram_size": 4},
}
model.generate(..., num_beams=4, length_penalty=1.2, do_sample=False,
               eos_token_id=EOS, **DECODE[tgt])
```

`scripts/decode_search.py` 可重跑解碼 grid；應用端一般只需對齊上表，不必重搜。
