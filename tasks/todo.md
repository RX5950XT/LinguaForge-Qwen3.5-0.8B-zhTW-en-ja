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
- [~] F24 新面板 v5a 三軸 — **執行中**（logs/eval/v5a-cap.log，25 篇 × 6 向 ~1h45m）
      doc 軸前 4 向已出：**截斷率全 0%**（v4 ja→en 是 8.3%）→ F3 長度外插確認修好；
      en→zhtw 文件級 22.99（v4-rp1.1 16.64 / base@12 20.74）、簡體洩漏 4.3% 守住 ≤5%
      ⚠️ v4/base 的 25 篇版**不重跑**：改判讀規則為看絕對健康指標（截斷/完整度/重複/洩漏），
      不跟 12 篇的舊數字硬比（那正是先前踩過的 n 混淆坑）
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
      CUDA illegal memory access）。**延到 v5b 訓練完後跟 v5b 一起掃**，
      一次拿到 base/v4/v5a/v5b 四版對齊；先跑它只會延後訓練 1.3 小時，且不影響 v5b 設計
- [x] **F29 找到 v5a 輸給 base 的真主因：語料語意錯位**（docs/RESEARCH-v5.md F7）
      `prepare_data` 清洗全是規則式，抓不到「兩句根本沒關係」。LaBSE 掃全 20 語料：
      globalvoices.ja-zhtw 64.7% / opensub.ja-zhtw 53.5% / **globalvoices.en-zhtw 42.6%** /
      coct.en-zhtw 19.9% 低於 0.60；對照 jparacrawl 0.8%、newscomm.ja-zh 1.5%。
      按實際配額加權後**四個 en-pivot 方向完全單調**：en→ja 7.8%→+3.82、ja→en 11.7%→+1.00、
      en→zhtw 17.1%→−1.44、zhtw→en 17.3%→−0.62。目檢確認是累積位移不是 LaBSE 假象
- [x] F30 `scripts/bitext_filter.py`（LaBSE 分數快取→`data/labse/*.npz`）＋
      `prepare_data.labse_filter()`，`LABSE_MIN = 0.60`。門檻用 FLORES 黃金對齊校準：
      三語言對 p01 都 ≥0.641，0.60 砍不到正確樣本。快取缺檔／涵蓋率<99% 一律硬報錯。
      套用在 `build_docs()` 前 → 錯位行留下行號缺口，文件重組自動斷開
- [ ] F31 **v5b 訓練** — ⏸ 阻塞中：使用者 2026-07-28 09:50 拆機換 RTX 5060 Ti 16GB，
      GPU 不在機上。硬體就緒後照 CONTEXT.md「裝好新卡後的接續步驟」直接跑。
      內容：F7 語意過濾 + F6 全形空白，其餘全部不動（20,000/方向、
      replay 池、LoRA 超參都跟 v5a 相同）→ 差異只歸因於資料品質。
      判讀：en→zhtw COMET ≥85.92 則 F7 成立出貨；84.48~85.92 則門檻不夠；≤84.48 證偽

## v5b → v5d（2026-07-29）

- [x] F31 v5b 訓練（F7 語意過濾 0.60 + F6 全形空白）。結論：方向對、力道不夠——
      六向平均 COMET +0.25、五向皆升，但 en→zhtw 84.97 仍低於 base。
      副產物教訓：`load_best_model_at_end` 依 eval_loss 挑 checkpoint 挑錯了
      （差 0.0003 的雜訊，實測 COMET 低 0.29）→ v5c 起關閉
- [x] F32 v5c 訓練（LaBSE 門檻 0.60→0.65）。F7 到頂：+0.17。F5 專有名詞意外治好。
      露出真病灶＝資料量，見 F35
- [x] F33 **解碼端定案**（docs/RESEARCH-v5.md F9/F11）：beam 4 修漏譯、
      `no_repeat_ngram_size=4` 修重複，逐目標語言分開設（en 不能用 nrng，
      合法 4-gram 重複多；zhtw 不能用 rep-penalty，F3 實測洩漏 4.65%→13.06%）。
      **零訓練成本買到 +0.92 COMET**，是 v5c 訓練 7h49m 所得（+0.17）的 5 倍
- [x] F34 **洩漏指標修正**（F10）：舊版 `s2tw` round-trip 判的是異體字偏好不是簡繁，
      灌水 2~4 倍。改成簡體專用字集 − 31 字台灣正字白名單，自檢
      `scripts/test_evaluate.py`。ja→zhtw 0.79% 達標；連帶推翻「v4→v5 洩漏退步」
- [x] D0 eval_loss 可比性（欠了很久）：`prepare_data.py --dev-from` 沿用既有 dev.jsonl
      並把其句對排除出 train。v5d 實測排除 1,194/1,200、train 內殘留 0
- [x] F35 **v5d 訓練完成**（07-29 23:18，3,412 步／2h28m 最後一段，死 3 次續跑）：
      每方向 40,000 × 1 epoch、272,783 筆。**eval_loss 1.831**（v5c 最佳 1.866，
      半個算力就贏）、FLORES chrF++/BLEU 十二格全勝 v5c-b4n4、**COMET AVG 86.34**
      （v5c 86.09）。假設成立：v5c 第二個 epoch 確實在背資料
- [x] F36 **v5d 通用能力抽查**：n=90 重新基準（題庫從 12 長到 90，舊 base/v3/v4 數字作廢）
      → base 78.9 / v5c 70.0 / **v5d 71.1**（+1.1，zh-TW +10.0，翻譯機率 4.4→2.2）。
      **沒有復發**，replay 佔比掉到 12.9% 不構成崩壞
- [x] F37 **結案：答案是資料量。** v5d en→zhtw **86.23 > base 86.19**，微調首度不再輸
      這個方向；同一趟洩漏 2.2%→1.19%。此前六種解釋全被排除（COMET 簡繁不敏感、
      指令措辭、長度比、絕對長度、第三 epoch、專有名詞）

## v5e → v5f（2026-07-30）

- [x] F39 **v5e 訓練完成**（07-30 11:30，5,677 步／10h30m，peak VRAM 4.00 GB）：
      `--limit 80000`、502,993 筆，其餘完全不動。**eval_loss 1.7708**（−0.060）、
      **COMET AVG 86.56**、六方向零退步（en→ja 88.44 與 zhtw→ja 87.14 為全系列新高）。
      同樣本數對照：step ~3077 看完 272,783 筆時 eval_loss ≈1.813 已低於 v5d 終值 1.831，
      且當時 LR 還很高 → **加量本身有效，不是 cosine 排程假象**
- [x] F40 **replay「佔比 vs 絕對量」定案**：三個觀測點 22.9%→12.9%→**7.0%**，
      絕對量恆為 35,177，通用能力反而 70.0→71.1→**72.2** 逐版上升。**佔比不是變因。**
      分語言 zh-TW 63.3（−6.7）與翻譯機率 3.3（+1.1）落在雜訊帶（每語言 n=30，
      一題＝3.3 個百分點，即 2 題與 1 題之差），不構成訊號
- [x] F41 **`--limit` 這條路走完了**：×1.77 換 +0.25、×1.84 換 +0.23，報酬持平偏降；
      且 **ja↔zhtw 首次抽不滿**（75,105／80,000，文件級只湊到 7,105）。再加量要先補來源
- [ ] F38 **en→zhtw 連兩版停在 86.23**（base 86.19），資料共加 ×3.6 只動 +0.04，
      其他五向從 base 起算都拿到 +0.5~+3.6。**新假設：瓶頸在目標側語域不在數量**
      —— zh-TW 目標主要來自 coct 與 opensub 字幕（1,034,818 筆），跟 FLORES 書面散文
      語域差距大，加同源語料只是把同一種語域加厚。**驗證需要新的 zh-TW 書面語來源**
- [ ] F42 **v5f 訓練：LoRA r 64→128 / alpha 128→256**，資料沿用 v5e 不動。
      理由：eval_loss 全程單調下降到最後一步、零過擬合訊號，而加量邊際報酬已收斂
      → 瓶頸轉向容量。epoch 是較弱的槓桿（v5c 兩 epoch/141k 得 1.866 vs
      v5d 一 epoch/273k 得 1.831）。peak VRAM 4.00/16 GB，容量翻倍有餘裕
- [ ] F43 **replay 池擴充**（欠 F40 一個對照組，也是 F38 的同一個瓶頸）：
      `REPLAY_SHARE = 0.35` 從 v4 起就沒被滿足過（v5e 目標 251,900、實得 35,177），
      三個來源已抽乾。要補得找 Apache-2.0 相容的非機翻 en/ja/zh-TW 指令資料
      （TaiwanChat CC-BY-NC、dolly 家族 CC-BY-SA 都衝突；**zh-TW 最難**）
- [ ] F44 `token_budget: 1450` 是給舊 3070 Ti 8GB 調的，5060 Ti 16GB 只吃 4 GB。
      提高能縮 wall-clock，但改變 effective batch size ⇒ **不可混進實驗變因**：
      必須同步等比降 `gradient_accumulation_steps`（例如 2900×4）維持每 optimizer step
      的 token 數不變，且先用 `bench_step.py` 實測再定案

## 引入公開知識／常識基準（2026-07-31）

起因：`eval_capability.py --axis general` 每語言只有 n=30 自建題，1 題 = 3.3pp，
只有 n=90 總分能當訊號，且題目是自己出的、無外部可比性。

- [x] B1 `scripts/eval_bench.py` ＋ `test_eval_bench.py`（自檢 6 項全過）。
      計分改成**選項輪轉 ×4**：單輪會被位置先驗污染（v5f 押同一字母 63.4%），
      比選項文字 logprob 則貼著 25% 隨機基準。三條路的實測見 lessons.md
- [x] B2 BELEBELE 900 題 × `zho_Hant` / `jpn_Jpan` / `eng_Latn`（CC-BY-SA-4.0）
- [x] B3 知識軸：TMMLU+（`ikala/tmmluplus`，MIT，66 科台灣考題，zh-TW 原生）
      ／`cais/mmlu`（MIT，en）／`openai/MMMLU` JA_JP（MIT，ja），各抽 900 題
      —— 非平行，**只有同語言內 base vs finetune 的 Δ 有效**，不得跨語言比
- [x] B5 修 `logits_to_keep=1`：峰值 VRAM 15.85 → 2.29 GB，分數不變
- [x] B4 base / v5d / v5e / v5f 四版全數跑完，結果見 F49
- [ ] B6 出貨標準已寫進 `CLAUDE.md`（硬閘＋目標＋停止規則），
      新增的兩項硬閘（BELEBELE ≥ base−3.0、doc 完整度 ≥ base−5%）尚未接進
      `regression_guard.py`，目前靠人工核對

汙染立場：對 Qwen3.5 預訓練而言沒有任何公開基準能證明未汙染，但本專案量的是
**同一份題目上 base → finetune 的差值**，汙染兩邊共有、相減抵消。
→ 絕對分數不得對外宣稱能力，只能用 Δ。

## v5f 結案與 v5g（2026-07-31）

- [x] F42 **v5f：LoRA r 64→128 / alpha 128→256**，資料沿用 v5e。
      COMET AVG 86.56 → **86.74**（六方向全升），eval_loss 1.7708 → **1.7440**，
      通用能力 72.2 → **73.3**，翻譯機率持平 3.3%。代價僅 +6.7% 時間、+0.77GB VRAM。
      → **容量的性價比比加資料高一個量級**，F41「加量已耗盡」對照成立
- [x] F45 en→zhtw 解凍：v5d/v5e 卡在 86.23 分毫不差，v5f 到 86.34，
      chrF++ +0.36 / BLEU +0.71 同步上升 → **F38（目標側語域假設）證據不足**，
      比較像 r=64 撐不起六方向
- [x] F48 **v5f 踩硬閘，不得出貨**（新基準第一次套用就攔下一版）。
      BELEBELE zh-TW 51.28（base 55.81，−4.53）、ja 43.94（base 51.78，−7.84），
      門檻 ≥ base−3.0 → FAIL ×2。知識軸三語也全低於 base。
      機制：v5f 在 zh-TW/ja 押同一選項 55.2%/56.9%（en 僅 28.6%）→ 中日文退回固定先驗。
      **出貨候選回到 v5e**（三語全 ≥ base）。COMET AVG +0.18 買不回這個代價
- [x] F49 **三點連線完成（03:29）：退化的變因是 r，不是資料量。**
      | BELEBELE | base | v5d(r64,273k) | v5e(r64,503k) | v5f(r128,503k) |
      |---|---|---|---|---|
      | zh-TW | 55.81 | 57.25 | 57.28 | **51.28** |
      | ja | 51.78 | 50.31 | 52.36 | **43.94** |
      | en | 57.83 | 68.08 | 62.81 | 65.19 |
      | AVG | 55.14 | **58.55** | 57.48 | **53.47** |
      知識軸同向：37.78 / 36.99 / 35.25（base 36.81）。
      **r=64 的兩版兩軸都 ≥ base，資料 ×1.84 沒有代價；只有 r=128 崩** → 斷崖不是斜坡。
      副產物：v5d 其實是通用能力最好的一版（兩軸皆最高），但 COMET 最低（86.34）
- [x] ~~F46 v5g：r 256 / alpha 512~~ **取消**（不是暫緩）。r=128 已越線，
      r=256 只會更深；可用容量區間確定是 **r ≤ 64**。中間點 r=96 未探，但優先度低於出貨
- [x] F47 **doc / ifeval 補跑完成**（05:55，2h24m，v5e 與 base-doc25 各 25 篇 × 6 向）。
      **v2/v3 的長文腰斬在 v5e 已消失**：行數比精準 1.000、尾段譯出比 0.851~0.924、
      腰斬率 0~4%、頭尾平坦（tail−head −0.10~+0.02）。對照 v3 是 0.103/91.7%。
      **意外**：腰斬的反而是 base 的反面——base 尾段超譯（tail 1.10~1.65、行數比 1.4~1.5），
      F18「翻完不停繼續生成」的老毛病在 base 身上，微調修掉了。
      ifeval v5e 48.9 vs base 53.3（zh-TW 46.7 vs 53.3 為主要缺口，差 4 題）
- [x] F50 **doc 硬閘改定義**（自己寫的閘自己改，理由記在此以備推翻）：
      原「完整度 ≥ base − 5%」用 `completeness_median`（整篇字元比），v5e 4/6 方向不過。
      該指標把「行文精簡／多吐垃圾行／尾段腰斬」混成一個數字，三者互相抵消——
      base 的 ja→zhtw 拿 1.188 是靠 tail 1.646 撐的，同格洩漏 79.55%、chrF++ 只有 7.40。
      改成絕對值兩條：`tail_ratio_median ≥ 0.80` 且 `truncated_pct ≤ 5%`。
      **falsification：新閘套 v3 照樣擋下**（completeness 0.103 → tail 趨近 0）。
      `eval_capability.score_doc` 加 `line_ratio_median` / `tail_ratio_median` 兩欄，
      `alignment()` 有自檢（腰斬 / 多吐行 / 完全一致三個 case）
- [~] F51 **打包出貨 v5e** — 進行中：
      - [x] LoRA adapter → `release/lora-v5e/`（r=64 / α=128 已從 adapter_config 核對）
      - [x] 合併 bf16 → `release/merged-bf16-v5e/`（1.7GB，三方向抽測乾淨無洩漏，
            `generation_config.json` 已帶 eos `[248044, 248046]`）
      - [ ] GGUF **卡住**：本機唯一的 llama.cpp 是 `AI-Factory/` 底下的 **b8189**，
            不只沒有 `--no-mtp`，連 Qwen3.5 的 BPE pre-tokenizer 都不認
            （`get_vocab_base_pre()` 無對應 hash）。要換較新的 llama.cpp；
            那是別的專案的 checkout，未動。`export_gguf.py` 已改成偵測 `--no-mtp`
            支援與否＋`--name` 參數化（原本 `NAME` 寫死 v3 會覆蓋舊檔）
      - [ ] 模型卡：等 F28 的 base 對照數字
- [~] F28 **base 用 `--full` ＋ 現行 `DECODE` 重跑** — 跑中（`by8a79kld`，06:11 啟動）。
      **這不是技術債，是硬閘的完整性問題**：寫模型卡時發現「六方向 COMET ≥ base」
      一直是拿 `base b4`（n=500、beam4、**無** per-language `no_repeat_ngram`）在比，
      而 v5d/v5e/v5f 都是 n=1012 + 完整 `DECODE`。**樣本數與解碼都不同。**
      而且 `base-b4-flores.json` 六個 `comet` 欄位全是 null，84.59 這個數字
      只活在 `docs/RESEARCH-v5.md` 的表格裡，沒有機器可讀的來源。
      重跑後要重新核對 v5e 這道閘，並回頭修 CONTEXT/RESEARCH 的 base 欄
