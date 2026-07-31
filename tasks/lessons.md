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
**但裝好之後這句警告仍然會印**——因為 `causal_conv1d` 也在同一個 `is_fast_path_available`
判斷裡，而它在 PyPI 上連一個 wheel 都沒有（要編就得補與 torch 同大版本的 CUDA toolkit）。
`modeling_qwen3_5.py:421-424` 是**逐個 op 各自 fallback**，缺 causal_conv1d 只讓 depthwise conv
退回 `nn.Conv1d`（cuDNN，本來就快）＋解碼每 token 多幾個 kernel launch，可以忽略。
**通則**：這種「所有加速件 all() 成一個 flag」的警告訊息會掩蓋「哪一件真的缺」。
別拿警告當結論——去讀那個 flag 的組成，再用 `is_xxx_available()` 逐項驗，
否則會像我一樣把 30% 的 GPU 使用率誤診成 fallback（真因是小模型 batch=4 解碼 launch-bound）。

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

**單步時間幾乎與 token 數無關**（~1.05s 固定成本：gradient checkpointing 重算 + 大量小 kernel
的 launch 開銷，Windows 尤重），所以裝不滿就是純浪費。
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

## 2026-07-29：`prepare_data.py` 的 `--limit 20000` 不能省

v5c 重建資料時直接跑 `uv run python scripts/prepare_data.py`（照 CLAUDE.md 常用指令那行），
結果每方向 128,657 筆、總量 773,972——是 v5a/v5b（20,000/方向、153,977 總量）的 5 倍。
`RECIPES` 的內建預算是 130,000/方向，`--limit` 是唯一把它壓到 20,000 的開關，
而 CLAUDE.md 的速查那行沒帶這個參數，只有 CONTEXT.md:445 記著完整指令。

代價：一次 13 分鐘的白工，以及差點用錯資料量開訓——那會讓 v5c−v5b 同時混進
「門檻 0.60→0.65」和「資料量 ×5」兩個變因，整輪實驗作廢。

**教訓**：兩份文件記同一條指令時，速查版本省掉的參數就是未來踩雷點。
已把 `--limit 20000` 補回 CLAUDE.md。跑資料建置後第一件事是核對
`results/data_stats.json` 的 `directions`，跟上一版對不上就是參數錯了，不要先開訓。

## 背景訓練會隨 Claude Code session 一起死（2026-07-29）

v5d 13:29 啟動，14:46 在 step 613 無聲停止。不是當機——`nvidia-smi` 乾淨、
沒有 traceback、log 停在最後一個進度條。原因是背景指令是 agent shell 的子孫，
session 結束時被連坐帶走。**背景任務通知會說 `No completion record was found`，
看到這句就要當成「可能還活著也可能死了」，一律去驗行程與 log 時間戳。**

同一天又死了兩次（16:22 step 736、之前 step 613），死因完全相同。試過三種脫離方式：

| 方式 | 結果 |
|---|---|
| `Invoke-CimMethod Win32_Process Create` | 秒死 `forrtl: error (200): window-CLOSE event` |
| `schtasks /sc once` | 載完資料、死在載模型，同一個錯誤 |
| `schtasks` + `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` | **啟動 9 秒內**同一個錯誤 |

numpy/MKL 帶的 Intel Fortran runtime 一收到 console CLOSE 事件就自我終止，
而脫離互動 session 的行程拿不到穩定 console。第三次特別值得記：
`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` 是 Intel 官方關閉那個 handler 的開關，
理論上正中根因，實測**完全無效**——schtasks 收掉 cmd console 的時機太早，
handler 還沒被那個變數繞過就已經觸發。別再花時間試這條。

**結論：不要跟 Windows 的 detach 機制搏鬥。** 靠 `save_steps` +
`train_sft.py --resume-from-checkpoint auto`，optimizer/scheduler/RNG/dataloader
位置全在 checkpoint 裡，接回去等同沒斷過，最多損失一個 save 週期。

既然斷線無法避免，就壓低單次代價：`save_steps` 300 → **100**（≈11 分鐘）。
但別忘了斷線的固定成本不只 checkpoint——**重跑要先花 ~6 分鐘重新 tokenize
272,704 筆**（`num_proc=1`，約 800 examples/s），所以實際代價是「6 分鐘 + 最多 100 步」。
`load_best_model_at_end=False`，所以 `save_steps` 不必是 `eval_steps` 的倍數。

## resume 時 `save_steps` 以 checkpoint 的 `trainer_state.json` 為準（2026-07-29）

上面那條「壓低斷線代價」差點白做：改完 `configs/sft_lora.yaml` 的 `save_steps: 100`
續跑，checkpoint 還是每 300 步一個。**transformers resume 會拿
`<checkpoint>/trainer_state.json` 的值覆蓋 CLI/config**，而且只印一行 warning 就過去：

```
The following arguments do not match the ones in the 'trainer_state.json' within
the checkpoint directory: save_steps: 100 (from args) != 300 (from trainer_state.json)
```

看到 warning 提到自己剛改的參數，很容易誤讀成「有讀到我的值」——其實是相反，
**寫在後面的 trainer_state 才是贏家**。要中途改就直接改 JSON：

```python
import json; p='outputs/sft-v5d/checkpoint-2100/trainer_state.json'
s=json.load(open(p,encoding='utf-8')); s['save_steps']=100
json.dump(s,open(p,'w',encoding='utf-8'),indent=2,ensure_ascii=False)
```

**驗證方式不是看 warning，是看下一個 checkpoint 編號真的落在新間隔上。**
沒核對這一步，代價是下一次斷線多賠 286 步（≈32 分鐘）。

另外：**Monitor 工具在這種場景不可靠**，同一輪被 teardown 掉兩次（狀態 `stopped`，
無事件）。要單一「跑完通知我」訊號，用 Bash `run_in_background` 配 `until` 迴圈，
不要用 `tail -f` 這種永不結束的 Monitor 指令。

順帶：長時間背景任務**不要用 `| tee`**。python 到 pipe 是塊緩衝（8KB），
中途完全看不到進度；直接 `> log 2>&1` 配 `PYTHONUNBUFFERED=1` 才即時。
（v5d 資料生成那次也踩到：`≥` 在 cp950 stdout 下炸掉，但因為緩衝，
錯誤訊息跟成功輸出一起在最後才吐出來，`tee` 還讓 exit code 變成 0。）

## 大 vocab：只要最後一格 logits 就一定要 `logits_to_keep=1`（2026-07-31）

`eval_bench.py` 第一版寫 `model(**inp).logits[:, -1, :]`，跑起來 VRAM 15.85/16.31 GB
貼著 OOM，使用者先看到才回報。原因是 `.logits` 會**先實體化整張**
`[batch, seq_len, 248064]`——batch 16 × 800 token × 248K × 2 bytes ≈ 6.3 GB，
然後才被切掉只留最後一格。

```python
logits = model(**inp, logits_to_keep=1).logits[:, -1, :]   # 2.29 GB
```

峰值 15.85 → **2.29 GB**，分數逐位相同（64.17/51.25/58.75）。
訓練沒踩到純粹是因為 `token_budget` 早就壓死了每個 micro-batch 的 token 數。
**教訓**：vocab 248K 的模型，任何 forward 都要先問「我需要幾格 logits」。

## 選擇題基準必須輪轉選項，否則量到的是位置先驗（2026-07-31）

用「比 A/B/C/D 四個 token 的 logits」計分，v5f 看起來比 v5e 掉 5 分。
但 `max_letter_share` 顯示 v5f 有 **63.4%** 押同一個字母（base 43%，隨機應為 ~27%）
——acc 被答案位置的先驗污染，分不出「掉知識」還是「偏好變強」。

試過的三條路：

| 計分法 | base 三語 | 判定 |
|---|---|---|
| 字母 logits，單輪 | 70 / 42.5 / 50 | 有訊號但被位置先驗污染 |
| 比選項文字 logprob（題面不提字母） | 30 / 27.5 / 37.5 | **貼著 25% 隨機基準**，選項是完整句子時 logprob 被表面形式主導 |
| 字母 logits + 選項輪轉 ×4 | 64.2 / 51.3 / 58.8 | **採用**：正解在每個位置各一次，先驗自動攤平 |

去偏後 v5e 其實**高於** base（BELEBELE 57.48 vs 55.14），先前「掉 5 分」是假象。
**教訓**：小模型跑選擇題，沒去偏的分數不可信；而且要順手記錄
「最常選的選項佔比」，那是唯一能看出計分被污染的訊號。

## 宣告某個對照無效之前，先確認自己讀對檔（2026-07-31）

看到 `results/capability/base.json` 只有 n=12，就對使用者說
「一直以來的『通用能力比 base 低 6.7』是拿 n=90 比 n=12，對照無效」。
錯了——n=90 的 base 存在 `base-n90.json`，數字 78.9 一直有效。

**教訓**：推翻既有結論比建立結論需要更多證據。宣告「先前的比較無效」前，
先 `ls` 該目錄看有沒有同名變體，不要只讀最直覺的那個檔名。

## 切段比對時，索引要用哪一邊的長度（2026-07-31）

判「文件級翻譯有沒有腰斬」時，我把每篇切成前中後三段比字元數，索引全用參考譯文的
行數 `m = len(ref_lines)` 算，然後同時去切 hypothesis：

```python
for lo, hi in [(0, m//3), (m//3, 2*m//3), (2*m//3, m)]:
    hc = sum(len(x) for x in hyp_lines[lo:hi])   # ← 錯
```

base 的行數是參考的 **1.4 倍**，`[2m/3 : m]` 在它身上取到的是中段，尾巴整段沒被算進去
→ 量出「base 尾段只剩 0.52~0.62」，我據此對使用者宣告「腰斬的是 base」。
用 `hyp_lines[lo:]` 取到底重算，真實數字是 **1.10~1.65**，結論整個翻轉：
base 不是腰斬，是尾段超譯。

**規則**：兩邊長度不一定相等時，切片的結尾用 `[lo:]` 而不是 `[lo:hi]`，
否則長度差會被默默吃掉。要比對齊的段落就先確認 `len(a) == len(b)`，不相等就別切段。

連帶的第二個教訓：**一個把多種病混在一起的指標，會讓病互相抵消。**
`completeness_median`（整篇字元比）同時受「行文精簡」「多吐垃圾行」「尾段腰斬」影響，
base 靠第二項補償第三項拿到 ~1.0 的漂亮分數。拆成 `line_ratio` 與 `tail_ratio`
兩個各自單義的指標後，兩邊的真實行為才顯現。訂硬閘時要先問：
**這個數字變差，可能的原因有幾種？超過一種就不能拿來當閘。**

## 對照組沒跟候選同條件，等於沒有對照組（2026-07-31）

「六方向 COMET ≥ base」這道硬閘，從 v5c 一路用到 v5f，比的是：

| | 樣本數 | 解碼 |
|---|---|---|
| base（`base b4`） | 500 | beam4，**無** per-language `no_repeat_ngram` |
| v5d / v5e / v5f | **1012** | beam4 **＋** 逐目標語言 nrng |

兩個維度都不同。照候選的條件重跑 base，en→zhtw 從 86.19 變成 **86.32**，
於是「v5d 86.23 首度贏過 base」直接翻轉成落後 0.09。整條 F37 結案作廢。

更糟的是**那個 86.19 沒有機器可讀的來源**：`base-b4-flores.json` 的六個 `comet`
欄位全是 `null`，數字只活在研究文件的表格裡。手抄進 Markdown 的數字沒有型別、
沒有 schema、不會因為條件改變而失效，之後每一版都在跟它比。

**規則**：
1. 換過解碼參數、換過 `--limit`、換過評測腳本之後，**對照組要跟著重跑**。
   對照組不是跑一次就永久有效的常數。
2. 任何進表格的數字都要有對應的 json 欄位。`null` 欄位配上文件裡的數字 = 這個數字
   不曉得是哪一輪、哪個設定跑出來的。
3. 判讀時先對 `n` 與 `decode_defaults` 兩個欄位，不一致就不要比。

## 沒有 CI 的點估計不能宣告勝負（2026-07-31）

同一件事的第二層：修好對照組之後，en→zhtw 是 −0.09。但 paired bootstrap
（n=1012、1000 次重抽、兩系統共用抽樣索引）給的是 **95% CI [−0.414, +0.225]**，跨 0。

也就是說**當初的「+0.04 領先」與更正後的「−0.09 落後」，兩個都在雜訊裡**——
真相是這個方向一直跟 base 打平。我用一個假訊號推翻了另一個假訊號。

COMET 的 `system_score` 只是 segment 分數的平均，0.1 上下的差在 n=1000 量級
本來就測不出來。工具已補：`tools/comet/paired_bootstrap.py`。
**規則：宣告某方向贏過或輸給對照組之前，先跑 CI；CI 跨 0 就寫「打平」。**

## llama.cpp 的「關 thinking」旗標換過，舊寫法會被靜默忽略（2026-07-31）

v3 模型卡教的是 `--chat-template-kwargs '{"enable_thinking":false}'`。
在現行 llama.cpp（實測 b9437）這個參數**還在、不會報錯，但對 thinking 沒有作用**——
思考內容會以雜訊前綴洩進譯文：

```
Q8_0  翻譯成繁體中文：The night market is crowded on weekends.
      -> A. 週末的夜市很熱鬧。          ← 多一個 "A. "
Q4_K_M 翻譯成日文：這家餐廳的牛肉麵非常好吃。
      -> お好みで、このラーメンは美味しい。  ← 多一個 "お好みで、"，還把牛肉麵翻成拉麵
```

一度誤判成量化損失，但 **Q8_0 與 Q4_K_M 都有**，量化損失不會這樣同時出現。
換成 `--reasoning off --reasoning-budget 0` 後，Q8_0 輸出與 bf16 合併版**逐字相同**。

**規則**：CLI 旗標「沒報錯」不等於「有生效」。行為不如預期時先確認旗標在**當前版本**
還有沒有被讀，`--help | grep` 一次就知道——這比從模型端找原因快得多。

## Windows 主控台會打壞 CJK 提示詞，看起來像模型壞掉（2026-07-31）

`llama-cli -p '翻譯成繁體中文：...'` 在 cp950 主控台下，模型收到的提示詞已經是亂碼
（回顯就看得出來：`½Ķ���c�餤��G`），輸出自然是垃圾。改成 `-f prompt.txt`
（UTF-8 檔）立刻正常。同一顆模型、同一組參數，差別只在輸入通道。

**規則**：CJK 進外部 CLI 一律走檔案，不走命令列參數。判斷輸出品質前，
先確認**輸入**沒被編碼吃掉——回顯的提示詞就是最便宜的檢查點。
