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

## 更正：「2B 微調淨負」其實大半是 QLoRA 4-bit 量化的鍋（NF4 2×2 實驗）

先前只用 bf16 評測就下「2B 微調淨負值」太快。補做 FLORES 2×2（{base,v3}×{bf16,NF4}）後拆解：
官方2B bf16 86.98 → base NF4 86.29（**量化 −0.69，主因**）→ v3 NF4 86.23（**同精度微調 −0.06≈中性**）。
即**微調本身沒破壞品質**，那 0.65 差距主要是被 8GB 逼用 4-bit QLoRA 訓練的量化損失。
另：4-bit 訓出的 adapter 疊到 bf16 base 反而 −0.65（adapter 校準給 4-bit，遇 16-bit 修正過頭）。
**通則**：① QLoRA 模型要「同精度」評測才公平——只用 bf16 評 NF4-trained adapter 會混入量化+錯配兩種假象；
② 下「微調沒用」結論前，先排除量化/精度 confound，最好能用全精度 LoRA 對照（本案受 8GB 所限沒做，
待雲端 bf16 訓 2B 才能真正判定 data-vs-scale）。已存 evaluate.py --nf4 供復現。
