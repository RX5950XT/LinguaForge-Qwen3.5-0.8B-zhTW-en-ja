# LinguaForge v3 todo

計畫全文：本機 plan 檔（未進版控）
紀律：**先擴評測（診斷）→ 多元化資料（治療）→ 重訓**。每步增益都要多領域基準 + 顯著性驗證。

## Phase A：評測擴充（診斷，最高優先）
- [x] A1 evaluate.py 泛化為 benchmark loader registry + `--benchmark`（flores/ntrex/wmt22/alt/tico19）— 5 loader 全驗過，22 方向-基準格
- [x] A2 zhtw 目標參考策略：FLORES/NTREX 原生繁體乾淨；WMT/ALT/TICO 簡體 s2twp（次要），主看 CometKiwi+leak
- [x] A3 tools/comet/kiwi.py：CometKiwi reference-free（gated，隔離環境，src+hyp）— 已建，執行時測 gated
- [x] A4 scripts/scoreboard.py：方向×基準矩陣 + 洩漏矩陣 + Δ（per-benchmark 不平均）— 已建+渲染驗過
- [x] A5 scripts/regression_guard.py：動態守 v2 底線 — 已建（compare.py 顯著性延到 Phase D 真有 v3 可比時建）
- [x] A6 v2 全面板 baseline（greedy，limit 500）完成 → COMET 全成功；scoreboard 診斷出爐（results/scoreboard.md）
  - ⚠️ CometKiwi 失敗：① gated 403（HF token 未接受授權）② 隔離環境 comet 版本不認得該模型名。待修（非阻塞，診斷用 COMET+leak）
  - ⏳ greedy-vs-beam、TICO 封存於進 Phase C 前處理
  - **診斷結論見 CONTEXT.md**：v2 對 NTREX 新聞領域確有過擬合（−2~6 COMET），但醫療/Wikinews 穩；WMT into-English 崩（源側解析弱）；洩漏修正全領域穩健

## Phase B：資料多元化（先過污染閘）
- [x] B1 污染閘：dump_eval_lines.py（5 基準全量 33,984 行含 s2twp 變體）→ prepare_data 任一側命中即丟、缺檔硬報錯；測試過（含 s2twp 後碰撞 case）
- [x] B2 抽樣改寫：MAX_SHARE=0.5 + 領域標籤/realized 計數入 data_stats.json；預算 130k/方向
- [x] B3 新來源（皆已下載+入配方）：GlobalVoices en/ja-zhtw（原生繁新聞，pre_globalvoices 濾碎片）、
      KDE4 en/ja-zhtw（原生繁 IT，pre_kde4 detok+濾佔位符）、KFTT 440k、OpenSub en-ja 2M v2024、
      OpenSub en-zhtw 4.77M、MTNT ja-en 6.5k。噪聲來源 ONE_WAY 只進 →en 源側（護洩漏+en→ja）。
      GNOME 已查證為空（78 對）不用；UM/MultiUN 降級不做（診斷未指向法律域）
- [x] B3-run 全量清洗完成：train 773,389 / dev 1,200；抽驗洩漏 0.00%（n=5000）；
      污染統計：v2 來源命中 ~9.5k/290 萬池（0.3%，多為短句通用語碰撞）→ **v2 沒實質背考卷，診斷結論不變**
- [~] B4 合成 zhtw→ja **延後**：zhtw→ja 已 130k 滿額且領域多元；等 2B 診斷判定「資料 vs 能力」再決定要不要花半天 GPU

## Phase C：訓練（0.8B 主線 + 2B QLoRA 並行）
- [x] C0 bench_step.py 擴充 --model/--r/--nf4；configs/sft_lora_v3.yaml（r64/a128+NEFTune5+1ep）
      與 configs/sft_qlora_2b.yaml（2B NF4）建立；train_sft.py 加 NF4 分支+neftune 接線
- [x] C0-run bench 完成：0.8B r64 7.18GB / 2B NF4 bs1×768 6.12GB，皆 <8GB 無 fallback
      （註：bench 每個 micro-step 都呼叫非 fused AdamW，絕對 tok/s 低估實際訓練 ~2.5×）
- [x] C1 0.8B 全量訓練完成（07-23 02:49）：2390 步 / 8h01m，final eval_loss **1.988**
      （500→2.131 / 1000→2.071 / 1500→2.017 / 2000→1.991 / 2390→1.988，單調下降）
      train_loss 2.124、peak VRAM 4.01GB、adapter → outputs/sft-v3
- [x] C2 2B QLoRA 訓練完成（07-23 20:20，17h15m/2390 步）：final eval_loss **1.818**
      （500→1.928 / 1000→1.860 / 1500→1.826 / 2000→1.818，尾段遞減收斂）
      train_loss 1.908、token acc 0.6184、peak VRAM 5.10GB、adapter → outputs/sft-2b-qlora
      → 2B 全程碾壓 0.8B（1.818 vs 1.988），容量優勢確立（惟 eval_loss≠翻譯品質，待 COMET 定奪）
- [x] 0.8B v3 評測完成（07-23 ~22:20，flores/ntrex/wmt22/alt，greedy limit500 + COMET）：
      **regression_guard 6 項全 PASS**。zhtw→ja（原最弱）FLORES 83.14→84.09（+0.95）跨三域一致升；
      en→ja 85.81→85.55（過 85.31 底線）；簡體洩漏全面降（en→zhtw −1.2、ja→zhtw ntrex −2.4/alt −2.0）
      → scoreboard 存 results/scoreboard.md（多數 Δ 在 CI 噪聲帶，穩健訊號＝zhtw→ja 升+洩漏降+守門過）
- [x] 2B v3 評測完成（07-24 ~00:50，flores/ntrex/wmt22/alt greedy limit500 + COMET）：
      22/22 格全勝 v2，每域 +1.8~2.1 COMET；但簡體洩漏 7.0/7.2 破 FLORES 硬閘（0.8B 過）
- [x] TICO 開封（勝者 2B）：en→zhtw +1.22 / zhtw→en +0.97，dev−sealed 差距 ~0.8 → 真泛化非背課本
- [x] s2twp 後處理實證：v3-2b 洩漏 7.0/7.2→3.4/3.4（過閘）、chrF++ 不變；s2t 版→0 證殘留是台灣變體非真簡體
- [x] **base 地板補跑（使用者要求，07-24 ~03:40）**：官方 instruct 零樣本 0.8B+2B 全 panel+COMET
      → **關鍵翻轉**：官方2B COMET 每基準都 > v3-2b；「官方2B+s2twp」COMET 88.21/洩漏5.4（過閘），
      7 個 zhtw 方向 6 個勝 v3-2b → **2B 微調淨負值，出貨應選「官方2B+s2twp」**（0.8B 微調仍正價值）
      詳見 lessons.md「微調前一定要先量官方模型零樣本地板」
- [ ] C3 解碼：greedy-vs-beam 於 v2 實測後定案；eos 必留

## Phase C：訓練（0.8B 主線 + 2B QLoRA 並行）
- [ ] C0 bench_step.py 重量頭空間（當前 TRL）
- [ ] C1 0.8B：r64/alpha128 + NEFTune=5 + 單一較廣 epoch
- [ ] C2 2B QLoRA：configs/sft_qlora_2b.yaml + train_sft.py NF4 分支；bench <8GB
- [ ] C3 解碼：beam=4/penalties（依 A6），保留 greedy fallback；eos 必留

## Phase D：評判與出貨
- [ ] D0 eval_loss 可比性修正：Phase B 重生了 dev.jsonl（新領域＋污染閘），v3 的 eval_loss
      與 v2 的 2.1634 不同分佈不可比 → 訓練結束後拿 v2 adapter 在**新 dev 集**跑一次 eval
      （單次僅 ~13s）取得同基準數字，再談「破 2.16 地板」
- [ ] D1 v3 兩線跑全 DEV 面板 + scoreboard Δ + regression_guard + 顯著性
- [ ] D2 最終候選開封 TICO 一次；dev-vs-sealed 差距判泛化
- [ ] D3 merge 勝出者 → 六方向抽測 → 更新 README/CONTEXT/lessons

## Review（2026-07-24）

**成果**：v3 兩線訓練+全 panel 評測完成，總表落 `results/scoreboard.md`（六模型×五基準）。
- **0.8B 微調正價值**確立：v3 洩漏最低、zhtw→ja 跨域升、守門全過。
- **關鍵翻轉**：官方 Qwen3.5-2B 零樣本 COMET 每基準 > v3-2b；出貨最佳＝**官方2B + s2twp**（零訓練，FLORES 88.21/洩漏5.4）。
- **NF4 2×2 更正**：「2B 微調淨負」大半是 4-bit 量化稅（−0.69），同精度微調≈中性（−0.06），非微調破壞。

**整理/修正**：
- 修 base2bs2tw-*.json 殘值（chrf++/leak 原沿用 base-2b，未隨 s2twp 重算）→ 已於 s2twp hyp 重算。
- 清空目錄 outputs/smoke、outputs/trial、data/clean。

**未做（等使用者拍板）**：出貨打包 / CPU GGUF 測試 / 雲端全精度 bf16 訓 2B（見 CONTEXT「下一步」）。
剩餘技術債：CometKiwi gated-403；全 DEV 面板(1012)+paired-bootstrap CI（現為 limit500 greedy 無 CI）。

## Phase E：發布打包（0.8B v3 開源；GitHub 程式碼／HF 大檔）— 2026-07-25
決策定案：資料集只放 recipe（不 re-host 語料）／研究報告 Markdown 進 repo／權重＝LoRA+GGUF+合併bf16。

**第 1 段（本機直接完成）✅ 2026-07-25**
- [x] E1 目錄骨架：建 `docs/`、`release/{lora-v3,merged-bf16,gguf,assets}`；`.gitignore` 加 `release/`
- [x] E2 `scripts/plot_loss.py`（--with matplotlib，不污染主環境）→ `release/assets/loss_curve.png`：train 2.044 / eval 1.988
- [x] E3 複製 v3 LoRA 到 `release/lora-v3/`（5 檔，不含 optimizer/checkpoint）
- [x] E4 `release/lora-v3/MODEL_CARD.md`：FLORES scoreboard、量化稅2×3、推論範例(eos=[248046,248044]+s2twp)、GGUF、授權
- [x] E5 `release/.gitattributes`：LFS 追 *.safetensors *.gguf *.png
- [x] E6 `docs/REPORT.md`：實驗全紀錄（摘要→動機→資料→評測→v1/v2/v3→多領域診斷→2B對照→量化稅→結論→復現）

**第 2 段（本機一次跑）✅**
- [x] E7 export_model.py（已參數化，不需改碼）--adapter sft-v3 → `release/merged-bf16/`（1.7GB）；三方向抽測乾淨無洩漏

**第 3 段（llama.cpp，實跑驗證）✅**
- [x] E8 `scripts/export_gguf.py`：convert_hf_to_gguf `--no-mtp` → llama-quantize Q8_0(775M)+Q4_K_M(505M)+f16(1.5G)
      **關鍵坑**：預設轉換把 MTP 併主檔 → runtime 報 `missing tensor blk.24.attn_norm.weight`；`--no-mtp` 修
      工具走官方預編 win 版（b10107，CPU+CUDA），非自編（msys ninja DLL 衝突放棄）
- [x] E9 GGUF 實測：**CPU 跑 ✅ 29 t/s、GPU(-ngl 99) 跑 ✅ 128–190 t/s**，皆正確台灣正體零洩漏；
      **MTP 可開 ✅**：`--mtp` 單獨匯出 head(496M) 當 speculative draft；
      注意 llama-cli 要 `--chat-template-kwargs '{"enable_thinking":false}'` 關 thinking 才乾淨
- [x] E10 更新 README/CONTEXT/AGENTS 目錄與發布說明

## Review（Phase E，2026-07-25）
**成果**：0.8B v3 開源打包全數完成，產物落 `release/`（gitignored）。
- LoRA(173M) + 合併bf16(1.7G) + GGUF(Q8_0 775M / Q4_K_M 505M / f16 1.5G / MTP 496M) + loss 曲線 + MODEL_CARD + REPORT。
- **GGUF 實測全過**：CPU 29 t/s、GPU(-ngl99) 128–190 t/s、MTP 可匯出當 draft；皆正確台灣正體零洩漏。
- 新腳本 `plot_loss.py`、`export_gguf.py`（固化 `--no-mtp` recipe）。
- git/HF 分流：`.gitignore` 加 `release/`；`.gitattributes` LFS 追大檔；語料不 re-host（recipe 復現）。

**踩坑**：① 預設 GGUF 轉換併 MTP → runtime 缺 `blk.24` → `--no-mtp`。② msys ninja 自編 DLL 衝突 → 改官方預編 b10107。
③ llama-cli 預設開 thinking 汙染輸出 → `--chat-template-kwargs '{"enable_thinking":false}'`。④ `-st` 必加否則卡等 stdin。

**待使用者**：實際上傳 HF（repo id + hf login）；是否也打包「官方2B+s2twp」；雲端 bf16 訓 2B。

## Phase F：v4 重訓（修災難性遺忘 + 長篇腰斬）— 2026-07-27

- [x] F1 三軸能力面板 `scripts/eval_capability.py`（general / ifeval / 長篇完整度＋腰斬率），
      洩漏率改逐「行」計（逐文件會直接飽和到 100%，也無法跟單句基準比）
- [x] F2 base vs v3 實測 → **v3 每一軸都退化**：→en 完整度 0.103/0.143、腰斬率 91.7%/66.7%
      （base zhtw2en 譯出 243 行，v3 只有 23 行）；ifeval 64.7→47.1；通用正確率 83.3%→33.3%
- [x] F3 replay 資料集 `scripts/build_replay.py`：en 16,554 / ja 17,170 / zhtw 1,453 = 35,177
- [x] F4 **JSONL 換行陷阱**修掉（`splitlines()` vs `json.dumps`）：寫入端 `LINESEP_RE`，
      讀取端 8 檔 12 處改 `rstrip("\n").split("\n")`，self-check 加回歸斷言
- [x] F5 TED2020 三項資料缺陷修掉：CJK 間空格（中文側 46% 的行）、句末缺標點（跨語言證據法
      `restore_pair`）、文件級併段時排除「兩側皆無標點」的行 → en-zhtw 保留 258,695 → 359,287
- [x] F6 文件級預算水位填補：小池子（newscomm 只有 59 篇）取不滿的餘額讓給大池子
      → 六方向全部補滿 3,000 篇（先前 en↔ja 只有 1,559）
- [x] F7 **packing 關掉**（sdpa 下跨樣本汙染，實測 logits 差 6.6250；Qwen3.5 linear attention
      更是修不掉）；超過 max_length 的樣本整筆丟棄而非截斷（截斷＝在示範講到一半停）
- [x] F8 **token 預算組批** `TokenBudgetSFTTrainer`：固定 bs2 只裝得下 ~250 token/micro-batch，
      硬上限是 1450 → 1.75 → 9.4 samples/s，48h → 7h
- [x] F9 v4 資料集：六方向各 20,000（各含 3,000 文件級）＋ replay 35,177 = 153,977（replay 22.85%）
- [x] F10 v4 訓練啟動：r64/α128、2 epoch、3,828 步、lr 1e-4 cosine、`load_best_model_at_end`
- [ ] F11 訓練完成後跑三軸面板 + 六方向翻譯基準，與 base / v3 比
- [ ] F12 繁中 replay 只有 1,453 筆（缺口）：授權可用的繁中指令集全是 CC-BY-NC 或未宣告授權，
      2B 蒸餾實測 18 組/分鐘（實測批次放大無效，瓶頸未定位；不是 linear attention fallback）
      → 補 6K 要 5.5 小時 GPU。**暫緩**：F14 顯示唯一支持「繁中 replay 不夠」的證據
      （zhtw ifeval 66.7→50.0）其實只差一題（n=6），不足以撐 5.5 小時的花費

## v5（診斷見 docs/RESEARCH-v5.md）

- [x] F13 **找到 en→zhtw 退步的真因**：`--limit 20000` × `MAX_SHARE 0.5` 讓依序貪婪的來源
      迴圈只餵得到前 2 個語料，每方向領域組合整個消失。反證是 en→ja 被砍一樣多的量
      （130k→20k）卻 +1.78，因為它前兩個來源剛好含 wikimatrix；en→zhtw 只有 textbook+news
      → −2.98。**不是先前判斷的「翻譯資料量砍太兇」**
- [x] F14 評測面板樣本數稽核：general n=12、ifeval n=17（zhtw 6 / ja 5 / en 6）。
      「zhtw ifeval 退步」＝ 4/6→3/6 差一題，不可用來做 v5 決策。只有 FLORES（1,012 句）有統計意義
- [x] F15 訓練/評測長度落差：v4 訓練樣本最長 581 token（`DOC_MAX=6` 卡住，`max_length 768`
      一筆都沒濾掉），評測文件句數中位數 11、p90 28，要生成到 2048 → 日文側 75~100% 貪婪迴圈
- [x] F16 修 `prepare_data.waterfill()`：注水式來源配額（句級＋文件級共用），
      自檢已加；模擬驗證 budget 20,000 下領域數 2~3 → 5~7，且完整預算下 en→ja 3→5、ja→en 4→7
- [x] F17 `DOC_MIN, DOC_MAX 3,6 → 4,16`；`max_length 768 → 1408`（configs/sft_lora.yaml，
      v4 版存成 sft_lora_v4.yaml）
- [x] F18 **迴圈分兩種病**（分桶 + base 對照）：非日文輸出是長度外插（分佈內 6% → ≥13 句 41%，
      `DOC_MAX` 治得了）；日文輸出在分佈內就 87.5% 迴圈，且 **base 自帶**（en→ja 6/12、
      zhtw→ja 3/12，其餘四方向全 0/12），我們的訓練把它放大（zhtw→ja 3→9→12）。
      實際長相是「翻完不停、開始自由生成、最後崩壞」，`im_end` 從未吐出。
      訓練資料端查無不對稱（三語各 39,600 筆、文件級皆 15.0%、收尾標點 4~6%）
- [x] F19 rep-penalty 1.1 全六方向跑完：**只換解碼參數，v4 就從六方向全輸 base 變成四方向贏**
      （en→ja 7.28→17.37 破 base 8.83、zhtw→ja 5.69→12.78 破 8.57、ja→zhtw 6.82→9.90 破 7.06、
      zhtw→en 45.81→52.72 破 50.63）。剩兩個輸的都不是重複造成（rep 已 8%/0%），
      對照 FLORES 切開病因：**ja→en 句級 COMET 贏 base +1.09、只有長文垮 → DOC_MAX 對症；
      en→zhtw 兩軸都退（−4.10/−2.98）→ F13 配方截斷，v5a 唯一的賭注**。
      **zh-TW 輸出不可開 rp**：洩漏 en→zhtw 4.65%→13.06%、ja→zhtw 0.56%→3.85%（base 是 68.26%）
- [x] F20 解碼預設依目標語言分流：`evaluate.DECODE` 當單一真相來源，eval_capability import 它，
      FLORES 與軸 A 都按目標語言帶入，結果 json 加記 decode_defaults。
      模型卡等 v5 打包時一起寫（現行 release/ 是 v3，不該塞 v4 才量到的數字）
- [x] F21 能力面板 ifeval 17→90、general 12→90（每語言各 30），
      加 scripts/test_eval_capability.py 夾兩邊（廢話不得通過、正確答案不得被擋）。
      抓到 9 題會被英文拒答 vacuously pass、`_sep_only` 放過帶前言的答案，都已修
- [x] F22 v5a 訓練完成（2026-07-28 06:20）：注水式配方 + DOC_MAX 16 + max_length 1408，
      每方向仍 20,000。4,468 steps / 8h30m / 峰值 4.00GB。train_loss 1.7462、eval_loss 2.0746
      **結果：只買到 en→zhtw COMET +1.36（83.12→84.48），其餘五向持平（−0.28~+0.28）。**
      沒過 85 門檻 → F13 只成立一半，配方截斷是主因之一但不是全部
- [~] F24 用新面板重跑 base 與 v4 的 ifeval/general — **執行中**（tasks/{v5a,v4,base}-cap.log）
      三軸各跑 v5a / v4-new / base-new，doc 軸 25 篇 × 6 向 × 3 模型，數小時
- [x] F25 en→zhtw 剩餘 1.44 定位到**專有名詞音譯**：句級 COMET 含專有名詞 −2.42 /
      不含 −1.00。訓練資料 95% 音譯（跟參考風格一致），但 0.8B 記不住名字 →「自信地猜」
      （`Dalhousie → 達爾西`）。猜錯比留英文更傷 COMET。**容量問題，無便宜資料側解法**
- [x] F26 修 `prepare_data` 全形標點後空白：`CJK_GAP_RE` 漏掉 U+3000-303F 與 U+FF01-FF60，
      訓練資料 16.3% 帶著它、模型輸出放大到 56~76%（參考 3%）。已補 + 自檢。
      ⚠️ 對指標幾乎沒差（chrF++ +0.02~0.10），是出貨品質不是分數
- [x] F27 掃訓練資料同類清洗漏洞：只有 F26 超過 5%，其餘（半形標點混用 0.81%、
      破折號殘留 1.20%、括號未配對 1.57%、全形英數 3.91%）都不值得單獨處理
- [x] ~~F23 擴量到每方向 60,000~~ **否決**：覆蓋率對資料量是對數關係（每翻倍 +8pp），
      擴 3 倍只讓 12.7% 的專有名詞句換桶 → 全體期望 **+0.14 chrF++**。21.6 小時不值得。
      F12 自蒸餾一併擱置（它只是為了配合擴量維持 replay 佔比）
- [ ] F28 base 用 `--full` ＋ 現行 `DECODE` 重跑 FLORES（前次與 segcomet 搶 GPU 撞
      CUDA illegal memory access）。這是「v5a 對 base 是否全面勝出」的最後一塊拼圖
