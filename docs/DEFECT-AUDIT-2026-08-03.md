# 下游缺陷稽核（2026-08-03）— A/B/C/F 是接入端 prompt，不是語料

VoiceInk（GGUF + node-llama-cpp）回報 30 句樣本 22 個客觀缺陷，量化三精度數字接近
（22 / 20 / 22）故判定「不是量化問題，是模型問題」。**前半正確、後半不成立。**

**結論：A（標籤前綴）、B（專名消失）、C（年份幻覺）、F（極短句退化）全部來自接入端
少餵 4 個 token 的 prompt。D（多行）與 E（領域術語）才是真的模型／語料問題。**

復現與驗收工具：`scripts/bench_defects.py`（transformers 側）、
VoiceInk `scripts/verify-chat-wrapper-fix.js` + `scripts/bench-cases.js`（GGUF 側）。

---

## 1. 第 0 步：缺陷復現

出貨 DECODE（beam=4 / lp=1.2 / 雙 EOS / 逐語言 rep+nrng），`release/merged-bf16-v5e`：

```powershell
uv run python scripts/bench_defects.py --label ship-beam4
uv run python scripts/bench_defects.py --label ship-greedy --beams 1
```

| 指標（30 句） | TF beam4 | TF greedy | GGUF f16 | GGUF Q8_0 | GGUF Q4_K_M |
|---|---|---|---|---|---|
| A 標籤前綴 | **0** | **0** | 9 | 9 | 8 |
| C 憑空年份 | **0** | 2 | 2 | 2 | 2 |
| B 拉丁專名保留 | 全保留 | H200→H20 | NVIDIA/H200/TSMC 消失 | 同 | 同 |
| D 多行 m1 | 失敗 | 失敗 | 失敗 | 失敗 | 失敗 |
| E `open weight` | 「開啟重量釋放」 | 同 | 「選擇重量釋放」 | 同 | 同 |

A 類**在 transformers 兩種解碼下都是 0**，連 greedy 都復現不出來。
所以不是「beam vs greedy」，是 transformers vs llama.cpp 這條路徑。

---

## 2. 根因：缺 `<think>\n\n</think>\n\n`

`release/merged-bf16-v5e/chat_template.jinja` L147–153：

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}     ← 未開 thinking 時固定輸出
    {%- endif %}
{%- endif %}
```

同一句提示詞的 token 化：

| 來源 | 尾端 token |
|---|---|
| transformers（訓練／評測／`evaluate.py`） | `… 248045, 74455, 198, **248068, 271, 248069, 271**` |
| node-llama-cpp 內建 `QwenChatWrapper` | `… 248045, 74455, 198` |

`248068/271/248069/271` = `<think>` `\n\n` `</think>` `\n\n`。
**模型從頭到尾是帶著這個空 think 區塊訓練與評測的**，接入端沒補就掉出分布。

已排除的其他變因（皆實測）：

- prompt 字串本身：chat wrapper 產出的字串與 `apply_chat_template` **逐字元相同**
- `dryRepeatPenalty`：關掉後 13 句輸出**一字不差**，不是它
- 量化：f16 也照樣發生
- `QwenChatWrapper` 的 `thoughts` 選項（`auto`/`discouraged`/`disabled`/…）：**六個值都不會補**，
  它是照 Qwen3 的 template 寫的，Qwen3.5 的 else 分支跟 Qwen3 不同

---

## 3. A/B 驗證：同 runtime、同 greedy，唯一變因是 think 前綴

f16 GGUF、`LlamaCompletion` 直餵 token（`tokenize(..., true)`）：

| 樣本 | 無 think 前綴 | 有 think 前綴 |
|---|---|---|
| n4 | 141GB HBM3e 記憶體。 | **NVIDIA H200** 擁有 141GB HBM3e 記憶體。 |
| n6 | **2009 年**，臺灣半導體產業將開始 2nm 的產量… | **TSMC** 今年年底將在新竹開始 2nm 的產量。 |
| n8 | **3.** 收入增長47%至3季度12億美元。 | 收入在第三季增長了47%，達到12億美元。 |
| p3 | **故事說，**豬會在晚上透過柵欄逃進森林。 | 豬會在晚上透過柵欄逃進森林。 |
| p4 | **1.** 病人應在飯後服用此藥兩次。 | 病人應在飯後服用此藥兩次。 |
| s3 | **選擇：**所有訂單免費運費 | 所有訂單免費運費 |
| s4 | **說明：**電池壽命： 18 小時 | 電池壽命： 18 小時 |
| c3 | **問：**您對GLM 5.5有什麼預測？ | 您對GLM 5.5有什麼預測？ |
| m2 | **這是個簡單的方法。**\n第一，煮水。… | 第一，煮水。\n第二，加入麵條。\n第三，等三分鐘。 |
| f1 | 大家好。 | 你好世界。 |

清單裡**每一個** A 類前綴都消失，B 類專名回來，C 類年份歸零。

### 生產路徑驗收（`LlamaChatSession` + VoiceInk 現行 `promptOptions`）

唯一變因是 `chatWrapper`：

```
node scripts/verify-chat-wrapper-fix.js [gguf]
```

| 指標 | Q4_K_M 修前 | Q4_K_M 修後 | Q8_0 修前 | Q8_0 修後 | 驗收門檻 |
|---|---|---|---|---|---|
| 有缺陷句 | 13 | **3** | 14 | **4** | — |
| 缺陷總數 | 20 | **5** | 20 | **6** | < 8 ✅ |
| A 標籤前綴 | 8 | **0** | 9 | **0** | = 0 ✅ |
| B 拉丁專名保留率 | 46.7% | 80.0% | 73.3% | **93.3%** | ≥ 90%（Q8 ✅／Q4 ✗） |
| C 憑空年份 | 2 | **0** | 2 | **0** | = 0 ✅ |
| D 行數遺失 | 1 | 1 | 1 | 1 | ≥95% 保留 ✗ |

> 缺陷總數與原始回報的 22 對不上，是因為**原指標的 `len` 門檻寫死 0.3~3**，
> 對跨語系不成立：英→中正常就落在 0.2~0.3，30 句裡有 12 句是這樣被誤判的
> （佔缺陷總數 55%）。已改成依語系分段（`bench-cases.js` `lengthRatioBounds`）。
> 用原門檻計分則是 22 → 16（Q4）。

Q4 的 80% 差在 n2 `Kimi`/`Sol` 被音譯成「金剛／索爾」，f16 與 Q8 都保留 → **這一項才是量化稅**。

### 修法（不是後處理）

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
new LlamaChatSession({ contextSequence, chatWrapper: new Qwen35ChatWrapper() })
```

llama-cli 對應做法是 `--jinja`（吃 GGUF 內建 template）；
`--reasoning off --reasoning-budget 0` 解的是另一件事（不解析 thinking），**補不了這 4 個 token**。

---

## 4. 語料稽核（deliverable 1）

`uv run python scripts/audit_corpus.py --samples 3` → `results/audit/corpus.json`
（`data/sft/train.jsonl`，468k 翻譯樣本 + 35,177 筆 replay 略過）

### 4.1 A 類：target 側標籤前綴（分方向）

| 樣式 | en→zhtw | ja→zhtw | zhtw→en | en→ja | ja→en | zhtw→ja |
|---|---|---|---|---|---|---|
| `enum:數字`（`^\d{1,2}[.、)]\s`） | **2,188 / 2.749%** | 7 / 0.009% | 15 | 14 | 14 | 3 |
| `label:問答`（`^(問\|答)：`） | **216 / 0.271%** | — | — | — | — | — |
| `label:註`（`^(註\|備註\|注意\|提示)：`） | 13 / 0.016% | 7 | — | 8 | — | — |
| `figure:圖為` | 9 / 0.011% | — | — | — | — | — |
| `narrate:據報導` | 2 | 3 | — | — | — | — |
| `figure:圖N` | 2 | — | — | — | — | — |
| `label:說明` | **1 / 0.001%** | — | — | — | — | — |
| `select:選擇：` | **0** | **0** | **0** | **0** | **0** | **0** |

### 4.2 A 類分來源（`data/raw/*.tsv`，中文側為 target，且 source 無對應標記）

| 來源 | 行數 | 主要污染 |
|---|---|---|
| **opus100.en-zh.tsv** | 982,180 | **`enum:數字` 77,356（7.876%）** ← 唯一高污染源，聯合國文件段落編號 |
| coct.en-zhtw.tsv | 310,290 | `label:問答` 754（0.243%）、`enum` 554、`圖為` 35 ← 光華雜誌訪談＋圖說 |
| globalvoices.en-zhtw.tsv | 136,153 | `label:問答` 148（0.109%）、`圖為` 37 |
| kde4.en-zhtw.tsv | 120,837 | `enum` 21、`註` 7、`說明` 3 |
| ted2020 / opensub / wikimatrix / newscomm | 5.9M | 全部 < 0.02% |

### 4.3 對使用者假設的判定

| 假設 | 判定 | 依據 |
|---|---|---|
| (1) 新聞圖說「圖為…」「圖 100 號：」 | **部分成立，量極小** | 全庫 11 筆（0.011%），且多數 source 本來就有 `The photo shows` |
| (2) 條列／百科行首 `1. ` `2. ` | **成立，且是最大宗** | opus100 7.876%、進 train 後 en→zhtw 2.749% |
| (3) UI 字串／選單「選擇…」 | **不成立** | 全庫零命中；kde4 是 UI 語料但無此樣式 |
| (4) 論壇 Q&A「問：」「答：」 | **成立** | coct 0.243%、進 train 216 筆；樣本確認是 source 無標記的憑空前綴 |
| (5) 譯文標註「譯者：」「說明：」 | **幾乎不成立** | 「說明：」全庫 3 筆、train 1 筆；「譯者：」零命中 |

**但這些污染在正確 prompt 下輸出率是 0。** 0.001%~2.7% 的低頻樣式只有在模型掉出
分布時才會浮上來——這正是缺 think 前綴做的事。

### 4.4 B 類：source 有拉丁專名、target 沒有

| 方向 | 寬版（含 `Japan` 這種該意譯的） | 嚴格版（縮寫／含數字型號） |
|---|---|---|
| en→zhtw | 48,859 / 56,803 = 86.01% | 4,984 / 8,008 = 62.24% |
| en→ja | 42,044 / 50,913 = 82.58% | 4,877 / 10,050 = 48.53% |
| zhtw→ja | 2,261 / 7,082 = 31.93% | 414 / 2,897 = 14.29% |
| ja→zhtw | 924 / 5,161 = 17.90% | 532 / 2,672 = 19.91% |
| zhtw→en | 448 / 6,859 = 6.53% | 380 / 2,969 = 12.80% |
| ja→en | 297 / 4,422 = 6.72% | 244 / 2,172 = 11.23% |

**兩個版本都不可直接當污染率。** 抽樣顯示大部分是正確意譯（`Japan`→日本、
`Mr. Ovia`→奧維亞先生）或誤報（把 `RT`、`II` 當型號）。真正該原樣保留的
產品名／型號在語料裡本來就稀少，這條指標**沒有可用的地板**——見 §6 補強計畫。

### 4.5 C 類：target 有四位數年份而 source 沒有

| 方向 | 筆數 | 比例 |
|---|---|---|
| zhtw→en | 782 | 0.982% |
| zhtw→ja | 436 | 0.584% |
| ja→zhtw | 401 | 0.537% |
| ja→en | 377 | 0.474% |
| en→ja | 359 | 0.451% |
| en→zhtw | 209 | 0.263% |

抽樣顯示成因是**雙語對齊錯位**（`1861年` ↔ `Created in 1862`、
甚至整句不對應），不是「新聞語料 target 普遍帶日期」。0.26–0.98% 屬於
bitext 挖掘的正常雜訊水位。

### 4.6 D 類：source 多行 → target 單行

| 方向 | 多行來源 | 壓扁 | 比例 |
|---|---|---|---|
| 六向全部 | 7,058 ~ 11,947 | **0** | **0.00%** |

**語料裡一筆都沒有。** 多行樣本全部來自 `build_docs()`，兩側都用 `\n` 對齊逐句併段，
結構天生守恆。m1 失敗的原因不是學到壓扁，是那三行（`Total parameters: 1T+` /
`Open weight release` / `Up to 1M context`）是**互不連貫的短標籤行**，
而訓練裡的多行樣本全是「同一場演講／同一篇評論的連續完整句」——分布外。

---

## 5. 清洗規則（deliverable 2）

只有 `opus100.en-zh` 的行首編號值得動；其餘樣式的量比誤傷風險還小。

| 樣式 | regex | 處置 | 誤傷風險 |
|---|---|---|---|
| 段落編號 | target `^\s*\d{1,2}[.、)]\s` **且** source 無 `^\s*\d{1,2}[.、)]\s` | **剝除**前綴，保留該句 | 低。已加 source 對照，`1. First step` → `1. 第一步` 不會被剝。剩餘風險是 source 用 `(1)`／`①` 而 target 用 `1.` → 建議把 source 判斷放寬到 `^\s*[(（]?\d{1,2}[)）.、]` |
| `問：`／`答：` | target `^(問\|答)[：:]` 且 source 無 `^(Q\|A\|問\|答)\s*[：:]` | **丟棄整筆**（僅 216 筆，剝了會留下語域不合的訪談口吻） | 中。`答：` 在真訪談逐字稿裡是合法譯文，但 source 側必然有 `A:`／`Answer:` → 已被 source 對照擋住 |
| `圖為`／`圖 N：` | target `^(圖為\|圖\s*\d*\s*[.號：:])` 且 source 無 `(The photo shows\|Shown here\|Pictured\|Fig(ure)?\.?\s*\d)` | **丟棄整筆**（11 筆） | 低 |
| `說明：`／`註：`／`注意：` | target `^(說明\|註\|備註\|注意\|提示)[：:]` 且 source 無 `(Note\|Warning\|Caution\|注意\|注記)\s*[：:]` | **丟棄整筆**（29 筆） | **中高**。`A warning: If you forget…` → `注意:…` 是**正確**譯文，必須靠 source 對照保住；規則裡的 source 白名單要含 `A warning`、`Caution`、`Important` |
| `選擇：` | — | **不做**。全庫零命中，寫了也是死碼 | — |
| `據報導，`／`相傳，` | target 命中且 source 無 `(It is said\|Legend has it\|Tradition has it\|According to report\|Reportedly)` | **丟棄**（6 筆） | 高。`Tradition has it that…` → `相傳，…` 是正確譯文；抽樣裡 3/4 都是這種 |

清洗總影響：約 **2,400 筆 / 468k = 0.51%**，其中 2,188 筆只是剝前綴不丟樣本。

**先講清楚：這些清洗改不了 30 句的任何一個數字**——A 類在正確 prompt 下已經是 0。
做它的理由是縮小「掉出分布時會浮出來的東西」，屬於保險，不是修復。

---

## 6. 資料補強（deliverable 3）

清洗補不回來的是 B / E，D 則要新增樣本形態。

| 類 | 缺什麼 | 新增樣本 | 估計筆數 |
|---|---|---|---|
| **B** 型號／產品名原樣保留 | 語料裡「拉丁縮寫 + 數字型號」句本來就稀少（en→zhtw 僅 8,008/79,602 含嚴格型名） | 科技新聞／規格表句對，source 含 `NVIDIA H200`、`GLM 5.5`、`TSMC 2nm`、`HBM3e` 這類 token，target 原樣保留 | 8k–12k（六向，per-direction 1.5k–2k） |
| **E** AI／科技領域術語 | `open weight`、`agentic coding`、`context window`、`inference` 等 2023 後詞彙不在 2020 前語料裡 | 術語對照句對（每詞 8–15 句不同語境，避免變查表） | 3k–5k |
| **D** 不連貫多行 | 多行樣本 100% 是連貫段落，沒有「獨立短標籤逐行」形態 | 規格表／清單／設定項逐行翻譯，行數嚴格 1:1，行內容互不相關 | 4k–6k |
| **F** 極短輸入 | `Hello world.` 這種 2–3 詞輸入在句級語料裡佔比極低 | tatoeba 已有短句，補「非句子」的名詞片語／招呼語 | 2k–3k |

合計 **17k–26k**，約現行 503k 的 3.4–5.2%。

**授權限制照舊**：CONTEXT.md 記錄 Apache-2.0 相容的 zh-TW 語料已抽乾。B/E/D 這幾類
沒有現成開源集，實務上只能靠「規則生成 + 人工抽檢」造（規格表／清單這類結構化文本
用模板生成不失真），這是**新語料工程，不是再跑一次訓練**。

---

## 7. 重訓配置與驗證集（deliverable 4）

**先不要重訓。** 理由：

1. A/B/C/F 已被 §3 的 4-token 修法解掉，重訓不會再改善它們
2. D/E 需要 §6 的新語料，語料沒到手前重訓等於重跑 v5e
3. `regression_guard.py --candidate v5e` 目前 exit 0，28 格全過；
   為 D/E 動 LoRA 有把 BELEBELE 洗掉的前科（v5f r=128 中日文 −4.53/−7.84）

語料到手後的配置（沿用 v5e，只動資料）：

```yaml
# configs/sft_lora_v6.yaml  ← 從 sft_lora_v5e.yaml 複製，不改超參
# LoRA r=64（r=128 已證偽）、一 epoch（v5c 兩 epoch 已證偽）
# 只加 §6 的 17k–26k 新樣本 + §5 的清洗，data 從 503k → ~520k
```

驗證集三層，缺一不可：

| 層 | 內容 | 門檻 |
|---|---|---|
| 硬閘 | `regression_guard.py --candidate v6` | exit 0（六條全過，含 COMET 三段判定） |
| 下游 | `bench_defects.py --label v6` + VoiceInk `verify-chat-wrapper-fix.js` | A=0、C=0、拉丁專名 ≥90%、多行 ≥95%、缺陷總數 <8 |
| 新增回歸 | 30 句樣本擴充到 60 句（B/E/D 各補 10 句），凍結後不再改 | 同上 |

第三層要先做：現行 30 句對 D 只有 2 個樣本（m1/m2），**分母 2 量不出 95%**。

---

## 8. 待辦

- [ ] VoiceInk 套用 `Qwen35ChatWrapper`，跑 `verify-chat-wrapper-fix.js` 存基線
- [ ] VoiceInk 下游的後處理 regex（`stripPromptLeak` / `LABEL_PREFIX` / `ENUM_PREFIX`）
      在修好 prompt 後應可縮減——但**先留著**，等 60 句樣本跑過再拆
- [ ] 30 → 60 句樣本擴充（D/E/B 各 +10）
- [ ] §5 清洗規則實作進 `prepare_data.py`（低優先，0.51% 影響）
- [ ] §6 新語料工程（阻塞 v6）
