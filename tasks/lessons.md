# Lessons

## 依賴衝突要隔離，不要遷就

`unbabel-comet` 鎖 `transformers<4.58`，但 Qwen3.5 的 `qwen3_5` 架構需要 5.x。
一開始裝在同一環境，導致模型根本載不起來。
**做法**：COMET 拆成 `tools/comet` 獨立 uv 子專案，透過檔案（`results/hyp/<tag>/*.txt`）交換資料。
**通則**：評測工具鏈與訓練工具鏈的依賴衝突很常見，第一時間就該隔離。

## Windows 上的 Qwen3.5 必裝 fast path

沒裝 `triton-windows` + `flash-linear-attention` 時，線性注意力退回
`torch_chunk_gated_delta_rule`，bs8×seq1024 直接分配 38GB 而 OOM——8GB 卡完全跑不動。
裝上後同樣設定只用 7.8GB。
**通則**：transformers 啟動時印的 "fast path is not available" 警告不是可忽略的雜訊，
對新架構模型可能是幾十倍的記憶體差異。

## 原始碼不要放字面控制字元

正規表達式裡直接寫零寬字元／控制字元，會讓 Python 因 null byte 拒絕讀檔，
且錯誤訊息（`source code string cannot contain null bytes`）指不到真正位置。
**做法**：一律用 `\uXXXX` escape。

## 想做 checkpoint 比較，就別讓 save_total_limit 刪掉它們

事後想比較早期 vs 最終 checkpoint 的分布外表現，卻發現 `save_total_limit=3` 早把
checkpoint-500/1000/1500 輪替刪光，只剩最後三個（2000/2500/2660）。
**通則**：若計畫要事後選 checkpoint 或畫分布外品質曲線，訓練前就要把 save_total_limit
設大或設 None，否則想驗證「是否過訓」時最有價值的早期權重已經沒了。

## 評測分數異常先看原始輸出，不要只看數字

微調後 zhtw→en 的 BLEU 從 19.4 崩到 2.44，數字上像「模型退步」。抽看 hyp 檔才發現
翻譯其實完全正確，只是模型在正確譯文後停不下來，一路吐 `user...assistant<think>` 到 256 上限，
BLEU 對這些垃圾懲罰極重。**根因是 EOS 不匹配**：SFT 資料以 `<|im_end|>`(248046) 收尾、
模型學會輸出它，但 base config 的 `eos_token_id` 是 `<|endoftext|>`(248044)，
`generate` 沒把 248046 當結束就失控續寫。修法：generate 明確傳
`eos_token_id=[248046, 248044]`（見 evaluate.py `stop_token_ids`）。
baseline 沒中招是因原始 instruct 模型較克制、剛好停得下來，微調強化了收尾行為才引爆。
**通則**：翻譯評測分數反常時，第一步永遠是讀模型的實際輸出，數字只是症狀。
**通則**：Qwen chat 模型推論務必確認 generate 的 eos 涵蓋 `<|im_end|>`，別信 config 預設。

## 「log 沒出現」不等於「不會出現」

訓練中途 `logs/train.log` 只有進度條，看不到任何 loss，我因此判定
「TRL 的 logging 不寫進 log 檔」並據此告知使用者監控無效。
實際上訓練結束時所有 log 一次 flush 出來，Monitor 完整收到 eval_loss 與 train_runtime。
正確的說法是「不即時寫入，process 結束才 flush」（tqdm/stderr 緩衝所致）。
**通則**：觀察不到某訊號時，先分清是「不存在」還是「還沒到」——
把緩衝行為誤判成功能缺失，會導出錯誤的架構結論並誤導他人。

## 資料要抽樣目檢，統計數字會騙人

首輪清洗後統計全綠（簡體殘留 0、各方向 75K 足額），但抽 6 句就看到
`(ヘレン) 本当にありがとう` → `HW:：謝謝你。謝謝你。` 這種字幕講者標記與對齊錯誤。
若直接訓練，模型會學會在譯文裡生講者標籤。
**通則**：每次資料管線改動後都要抽樣目檢，統計只能證明「沒有你已經在查的問題」。

## Windows 的 GPU 記憶體 fallback 會靜默吃掉 5 倍效能

超過 VRAM 時 Windows 的 NVIDIA 驅動不會 OOM，而是靜默 fallback 到系統記憶體。
第一次全量訓練跑了 **4 小時才 30 步**，`nvidia-smi` 全程顯示 GPU 100%、7.8GB——
看起來完全正常，實際上溢出到系統記憶體，速度只有 1/5。
**做法**：寫 `scripts/bench_step.py` 直接量單步時間與 `max_memory_allocated`，
一眼看出 bs4×seq1024 要 15.3GB。掃描後選出 8GB 內的最佳點 bs2×seq768。
**通則**：GPU 使用率高 ≠ 跑得快。吞吐量（tok/s）才是指標，且要先量再開長跑。

## 長跑前先量單步成本

原本直接啟動全量訓練，四小時後才發現配置有問題，白燒一個下午。
花 10 分鐘寫 benchmark 掃描配置，就能在開跑前確定「2660 步 × 17.9s = 13.2 小時」。
**通則**：任何超過一小時的任務，開跑前必須能回答「單位工作要多久、總共幾個單位」。

## 估算要用實測，不要憑感覺

估訓練時間時猜「每樣本約 140 tokens」，實測只有 71.8——整整差一倍，
連帶把 27 小時的錯誤估算修正為 12.7 小時，也改變了「要不要縮減資料量」的決策。
**通則**：影響決策的數字要實際量（`scripts/count_tokens.py`），不要用直覺代替。

## 別把資料品質問題誤判成「固有 trade-off」

v1 微調後 COMET 均分從 baseline 83.81 掉到 83.06，我測了過訓假設（否證）後便下結論：
「這是翻譯特化 vs 通用能力的固有 trade-off，非 bug」。**這個結論是錯的。**
真正病灶是 en↔ja、zhtw↔ja 全靠雜訊多的 OPUS-100（隨機網路/字幕、口語、對齊差），
而 FLORES 測正式書面日文——domain 不匹配。把 en-ja 換成 WikiMatrix/TED2020/JParaCrawl-filtered/
Tatoeba 等乾淨書面語料重訓（v2），COMET 均分升到 84.58，不僅補回 regression 還反超 baseline
+0.77，en→ja 更從 82.9 衝到 85.81（+2.9）。簡體洩漏修正全程保留。
**通則**：把某方向的退步歸因於「本質上就得犧牲」之前，先窮盡資料品質假設——
量一下該方向的語料來源與 domain 是否對得上測試集。過訓否證了 ≠ 資料沒問題。

## export 的 verify print 在 Windows console 會因 cp950 崩（但模型已存好）

`export_model.py` 合併後印日文譯文驗證，遇長音符 `ー`(ー) 觸發 `UnicodeEncodeError: cp950`，
process 回 exit 1 看似合併失敗——實際模型已在 print 前存好，純屬印出崩潰。
**做法**：腳本開頭 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`；
或用 `verify_merged.py` 寫 UTF-8 檔驗證（繞過 console）。
**通則**：Windows 上任何會印非 ASCII 的腳本都先把 stdout 轉 UTF-8，否則收尾 print 會誤報整體失敗。

## 先量基線再動手

基線顯示 COMET 82–86（語意沒問題）但 chrF++ 11–20（表面形式差）、簡體洩漏最高 47%。
這直接界定了微調該攻什麼——若沒先量，很容易誤以為模型「不會翻譯」而選錯策略。

## 微調前一定要先量「官方模型零樣本」地板（v3 最大教訓）

v3 花大量算力訓 0.8B + 2B QLoRA，最後補跑官方 instruct 零樣本才發現：
**官方 Qwen3.5-2B 零樣本 COMET 每個基準都比我們微調的 v3-2b 高**（flores 86.98 vs 86.33），
我們的 2B LoRA 反而小幅拉低 COMET，只換到簡體洩漏 16–24%→7%。而洩漏本可用既有的
s2twp 後處理零成本修掉——實測「官方2B+s2twp」COMET 88.21、洩漏 5.4%（過閘），
在 7 個 zhtw 方向 6 個贏過 v3-2b。結論：**2B 微調是淨負值，該出貨的是「官方2B+s2twp」**。
（對照：0.8B 上微調仍有正價值，COMET 小升+洩漏大降，因為 0.8B 官方較弱有空間。）
**通則**：任何微調專案第一步就要建「官方模型零樣本 + 廉價後處理」的地板，否則會把
「大模型本來就強」誤記為「我們微調的功勞」，白花算力。COMET 對簡繁不敏感，會獎勵
洩漏簡體的流暢輸出——單看 COMET 漏判洩漏，單看洩漏漏判品質，兩者必須並看。

## OpenCC `s2t` 不能拿來偵測簡體洩漏，要用 `s2tw`（2026-07-27，量測 bug）

專案從頭到尾用 `cc_s2t.convert(s) != s` 當「這句有簡體」的判據，錯了。
`s2t` 的目標是「傳統中文」而非「台灣標準」，它會做這些轉換：

| 原文（正確台灣用字） | s2t 轉成 | s2tw |
|---|---|---|
| 剛才 | 剛**纔** | 不變 ✅ |
| 人群 | 人**羣** | 不變 ✅ |
| 稽核 | 稽**覈** | 不變 ✅ |
| 软件和网络（真簡體） | 軟件和網絡 | 也會轉 ✅ |

於是任何含「剛才 / 人群 / 稽核」的正確譯文都被記成洩漏。用 `scripts/rescore_leak.py`
以 `s2tw` 重算既有 `results/hyp/` 全部譯文後，**60 筆分數被修正**，最誇張的是出貨組合：

| tag | 方向 | 舊(s2t) | 新(s2tw) |
|---|---|---|---|
| base2bs2tw-flores | en→zhtw | 5.40% | **0.20%** |
| base2bs2tw-flores | ja→zhtw | 4.00% | **0.00%** |

即「官方2B + s2twp」實際上幾乎零洩漏，不是 CONTEXT 原記的 5.4%。其餘 tag 多被灌水 1~3pp。
**通則**：字形正確性的判據要指定「目標地區標準」（s2tw / s2twp），不能用泛用的 s2t；
把工具的預設語義當成自己要的語義，會做出一個從頭錯到尾、卻看起來很合理的指標。
**通則**：譯文（hyp）一定要存檔。這次能零成本重算 60 筆歷史分數，就是因為
`results/hyp/` 留著；只存分數的話這個 bug 只能整批重跑模型才修得掉。

## 「純任務特化、不混通用資料」＝訂購一場災難性遺忘（v3 最大教訓，2026-07-27）

專案「已確認決策」寫的是「純翻譯特化（不混通用資料）」，訓練集 100% 翻譯任務、
0% 通用指令、100% 單句對、0% 多段落。五個基準 COMET 84~86 全部亮綠燈，
但手測發現模型**已經不是翻譯模型，是翻譯函數**：

- 問「台灣最高山是哪一座？」→ 模型把問題翻成繁中丟回來，不回答
- 「寫一個 Python 函數…」→ 把需求翻成英文
- 5 段 1491 字的文章 → 只翻前 2 段就發 `<|im_end|>` 收尾；v2 更會無限重複同一句

v3(r64/α128/1ep) 與 v2(r32/α64/2ep) 崩得一樣 → **不是 rank / epoch 的問題，是配比的問題**。
對照 SOTA：Tower+（Unbabel 2025）的 SFT 是 **22% 翻譯 / 78% 通用指令**；
文獻另載只 replay 1% 歷史指令資料就足以守住指令跟隨。我們用了 0%。

**通則**：任務特化微調一定要混通用 replay，比例寧高勿低（30% 起跳）。
「不混通用資料」不是省事，是主動把 base 模型的指令跟隨先驗覆蓋掉。
**通則**：訓練樣本的「形狀」會被學走。全單句對 → 模型學會一句就收尾，長文必漏譯。
要支援文件級輸出，訓練集就必須有文件級樣本。

## 基準測不到的能力，等於你沒有那個能力（同上事故的評測教訓）

FLORES / NTREX / WMT22 / ALT / TICO-19 五個基準都通過，COMET 84.8 也是真的——
但它們**全是單句、全是翻譯任務**，所以「長篇漏譯」與「通用能力歸零」這兩個
致命缺陷一個都沒被抓到。缺陷是使用者手動試用時憑感覺發現的，不是評測發現的。
**通則**：評測面板的覆蓋範圍就是你能宣稱的範圍。宣稱「模型可用」之前，先列出
實際 use case，再檢查每一項是否有對應指標；沒有指標的維度要當成「未知」而非「沒問題」。
**做法**：`scripts/eval_capability.py` 補三軸——文件級翻譯（含完整度/重複率，不只 chrF++）、
可驗證指令跟隨、通用能力保留（含「退化成翻譯機」自動偵測）。

## 更正：「2B 微調淨負」其實大半是 QLoRA 4-bit 量化的鍋（NF4 2×2 實驗）

先前只用 bf16 評測就下「2B 微調淨負值」太快。補做 FLORES 2×2（{base,v3}×{bf16,NF4}）後拆解：
官方2B bf16 86.98 → base NF4 86.29（**量化 −0.69，主因**）→ v3 NF4 86.23（**同精度微調 −0.06≈中性**）。
即**微調本身沒破壞品質**，那 0.65 差距主要是被 8GB 逼用 4-bit QLoRA 訓練的量化損失。
另：4-bit 訓出的 adapter 疊到 bf16 base 反而 −0.65（adapter 校準給 4-bit，遇 16-bit 修正過頭）。
**通則**：① QLoRA 模型要「同精度」評測才公平——只用 bf16 評 NF4-trained adapter 會混入量化+錯配兩種假象；
② 下「微調沒用」結論前，先排除量化/精度 confound，最好能用全精度 LoRA 對照（本案受 8GB 所限沒做，
待雲端 bf16 訓 2B 才能真正判定 data-vs-scale）。已存 evaluate.py --nf4 供復現。

## `str.splitlines()` 不等於「按 \n 切」——JSONL 會被切壞（v4 資料管線事故）

`prepare_data.py --limit 5000` 讀 replay.jsonl 時 `JSONDecodeError: Unterminated string`。
原因：Python 的 `str.splitlines()` 除了 `\n` 還會在 `\x0b \x0c \x1c \x1d \x1e \x85 U+2028 U+2029`
斷行，而 `json.dumps` **不跳脫**這些字元。35,083 筆記錄被讀成 35,093 行，
其中一筆 oasst2 回覆含 10 個 U+2028 → 被切成 11 段，第一段就是壞掉的半截 JSON。
**通則**：讀 JSONL / 逐行對齊的語料一律 `text.rstrip("\n").split("\n")`，不要用 `splitlines()`。
危害不只是炸掉——平行語料（`download_data.py` 的 `zip(lines1, lines2)`）只要有一側含
U+2028 就會**靜默錯位**，之後每一行的譯文都對到錯的原文，比直接崩潰更難發現。
**做法**：寫入端在清洗時把這些字元換成真 `\n`（`build_replay.py:LINESEP_RE`），
讀取端全面改掉 `splitlines()`（本專案 8 個檔、12 處），self-check 加回歸斷言。

## packing + sdpa = 跨樣本汙染，v2/v3 都中了（v4 訓練前查出）

TRL 的 `packing=True` 預設策略 `bfd` 會**自動啟用 padding-free**，而 padding-free
只有 FlashAttention 2/3 支援。本專案在 Windows 用 sdpa，等於把多筆樣本接成一條長序列
餵進去、只有一般 causal mask —— 後面的樣本看得到前面不相干樣本的全部內容。
TRL 原始碼裡有明確警告（`may lead to cross-contamination between samples`），
但那是 `logger.warning`，被訓練 log 淹沒沒人看到。

實測驗證（改前一個樣本的 token，看後一個樣本的 logits 變不變）：
```
改動前一個樣本後，後一個樣本 logits 最大差異 = 6.6250   → 汙染確認
```
這正好會教出「一段講完可以直接接上不相干內容」，與使用者回報的
「翻譯亂七八糟、牛頭不對馬嘴」高度吻合。

**Qwen3.5 更不能打包**：混合 linear attention 的遞迴狀態是沿序列累積的，
不受 attention mask 控制，樣本邊界根本切不斷——就算裝了 flash-attn 也修不掉。

**通則**：① 開 packing 前先確認 attention 實作真的支援樣本邊界隔離，別信預設值；
② 用「改 A 看 B 的 logits 會不會動」這招可以 30 秒驗證，不必讀框架原始碼；
③ 速度與記憶體只取決於「每步總 token 數」（bs1×1536 / bs2×768 / bs4×384 實測都是
6.93GB、~1400 tok/s），所以關掉 packing 改長度分組**不會變慢**，沒有理由冒這個險。

## 關掉 packing 後真正的代價是「micro-batch 裝不滿」，不是每 token 變慢（v4 訓練前實測）

上一則結尾寫「關掉 packing 改長度分組不會變慢」——**只對了一半**，要補正。
每 token 成本確實只看「每個 micro-batch 的總 token 數」，但固定 `batch_size` 必須
遷就**最長**樣本（max_length 768 → bs 只能開 2），而資料中位數只有 88 token，
於是絕大多數 micro-batch 只裝了 ~250 token，離 1450 的硬上限差了 6 倍。

實測（3070 Ti 8GB，r=64 + gradient checkpointing）：

| micro-batch | tok/s | VRAM |
|---|---|---|
| bs2×128（256 tok，實際訓練的常態） | 300~370 | 2.97GB |
| bs11×132 / bs4×362 / bs1×1450（~1450 tok） | 1300~1600 | 6.90GB |
| bs2×768（1536 tok） | 1230~1650 | 7.18GB |

**單步時間幾乎與 token 數無關**（~1.05s 固定成本：checkpointing 重算 + linear attention
torch fallback + Windows kernel launch），所以裝不滿就是純浪費。
實跑驗證：固定 bs2 是 1.75 samples/s，換成 token 預算組批後 9.4 samples/s，**5.4 倍**；
整個 v4 訓練從 48 小時降到 7 小時。

**修法**：`train_sft.py:TokenBudgetSFTTrainer` 覆寫 `get_train_dataloader`，
依長度排序後貪婪填批（`len(batch) * max_len <= 1450`），用 `batch_sampler` 餵 DataLoader，
每個 epoch 重洗批次順序。batch size 因此是動態的（平均 10 筆/批）。

**通則**：① vocab 大的模型（此處 248K）記憶體被 logits 支配（~3.3MB/token），
「每個 micro-batch 幾個 token」才是唯一該調的旋鈕，batch size 和 seq len 都只是它的因式；
② 關掉 packing 後一定要換成 token 預算組批，否則等於用最長樣本的規格跑完整個資料集；
③ transformers 5.x 拿掉了 `TrainingArguments.group_by_length`，TRL 1.8 改叫
`train_sampling_strategy='group_by_length'`——但它仍是固定 batch size，救不了這個問題。
