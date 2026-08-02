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

---

## 2. 已知怪行為（設定 vs 本質）

### 設定用錯就會出現

| 症狀 | 常見根因 | 修法 |
|---|---|---|
| 譯完後無限灌水到 max tokens | 只設單一 eos | 雙 EOS 如上 |
| 長口語／社群文→en（或 greedy→ja）同一句刷屏 | 未開 `no_repeat_ngram_size`；或 GGUF greedy | transformers 走 DECODE；GGUF 加 `--repeat-penalty` + 分段 |
| 譯文前綴出現 `A. `、思考句、日文雜訊 | thinking 沒關 | llama-cli：`--reasoning off --reasoning-budget 0`（舊 kwargs 會被靜默忽略） |
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

1) 雙 EOS（最重要）
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
   - 必關 thinking：--reasoning off --reasoning-budget 0（舊 enable_thinking 可能被靜默忽略）。
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

## 驗收（修完必跑）
A. 短句：The patient should take this medication twice a day. → 繁中，應為完整一句、無簡體、無灌水。
B. 短句：週末的夜市人聲鼎沸。 → 英文，一句、無重複尾巴。
C. 長口語／社群風多句中文 → 英文：不得同一子句無限重複；允許分段。
D. 生成長度：正常句不得貼滿 max_new_tokens（否則 EOS 未生效）。
E. 繁中輸出：抽樣無簡體專用字（或 s2twp 後為 0）。

## 實作要求
- 先定位應用裡 generate / llama 呼叫點，對照上表 diff，最小改動修到過驗收。
- 把 DECODE 常數集中成一處（依 target lang 查表），避免三處複製不一致。
- 修完用實際 log 證明：打印 eos_token_id、num_beams、length_penalty、no_repeat_ngram_size、是否 s2twp。
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
