# CONTEXT — 開發紀錄交接

> 給下一個 AI Agent：先讀 CLAUDE.md 的環境與指令，再讀本檔進度。

## ⚠️ v3 最終定案（2026-07-24）— 讀這段再動手

v3 訓完 0.8B(sft-v3) + 2B QLoRA(sft-2b-qlora)，五模型全 panel(FLORES/NTREX/WMT22/ALT)+COMET+TICO 評完。
**關鍵翻轉（補跑官方零樣本地板才發現）**：

| 基準 | 官方0.8B | 官方2B | v2 | v3-0.8B | v3-2B |
|---|---|---|---|---|---|
| flores | 83.85 | **86.98** | 84.58 | 84.80 | 86.33 |
| ntrex | 79.97 | **84.10** | 81.45 | 81.42 | 83.42 |
| wmt22 | 80.85 | **83.96** | 80.58 | 80.97 | 82.72 |
| alt | 82.81 | **86.66** | 84.48 | 84.94 | 86.31 |

- **官方 Qwen3.5-2B 零樣本 COMET 每基準都 > 我們微調的 v3-2B**；洩漏用既有 s2twp 後處理即可修：
  **「官方2B+s2twp」COMET 88.21、洩漏5.4%（過閘）** → **出貨建議：官方 Qwen3.5-2B + s2twp 輸出後處理，零訓練**。
- **⚠️ 更正（NF4 2×2 實驗）**：先前「2B 微調淨負」太快。FLORES 2×2({base,v3}×{bf16,NF4}) 拆解：
  base bf16 86.98 →(量化−0.69)→ base NF4 86.29 →(同精度微調−0.06≈中性)→ v3 NF4 86.23。
  **微調本身沒破壞品質，那 0.65 差距主因是被 8GB 逼用 4-bit QLoRA 的量化損失**（非微調的錯）。
  4-bit 訓的 adapter 疊 bf16 base 反而−0.65（校準錯配）。→ 真正判定 2B 微調價值需**全精度 bf16 訓 2B**（雲端 >8GB，未做）。
  evaluate.py 已加 --nf4 供同精度復現。出貨結論不變（base+s2twp 最佳），但「微調沒用」被否證。
- **0.8B 上微調仍有正價值**（COMET 小升 + 洩漏 20/47%→5/6%）；若要輕量部署才用 sft-v3。
- COMET 對簡繁不敏感（獎勵洩漏簡體的流暢輸出）→ **COMET 與洩漏率必須並看**。
- **完整分數表：`results/scoreboard.md`**（六模型 × 五基準 COMET + 洩漏率，由 results/*.json 自動彙整）。
- **資料修正（2026-07-24）**：base2bs2tw-*.json 的 `chrf++`/`simplified_leak_pct` 原沿用自 base-2b 未重算
  （s2twp 只重算了 COMET）；已於 s2twp hyp 上重算修正，真實 FLORES 洩漏 en→zhtw **5.4**／ja→zhtw **4.0**。
- 合成 zhtw→ja(B4) 免了（2B 診斷判定該方向是能力受限、官方2B 已解）。

## Phase E 完成：0.8B v3 開源打包（2026-07-25）

**決策定案**：資料集只放 recipe（不 re-host 語料，法律風險）／研究報告 Markdown 進 repo／權重＝LoRA+GGUF+合併bf16。
GitHub 留核心程式碼可複現；HF 放大檔（`release/` 已 gitignored）。

**產物（`release/`，皆 gitignored）**：
- `lora-v3/`：v3 adapter（173MB）+ tokenizer + `MODEL_CARD.md`（FLORES scoreboard、量化稅2×3、推論範例、授權）
- `merged-bf16/`：合併全模型 1.7GB（`export_model.py --adapter outputs/sft-v3`，三方向抽測乾淨）
- `gguf/`：**Q8_0 775M / Q4_K_M 505M / f16 1.5G / mtp-f16 496M**
- `assets/loss_curve.png`：train(119)+eval(5) 雙曲線（eval 1.988 破 v2 地板）
- `docs/REPORT.md`：實驗全紀錄；新腳本 `plot_loss.py`、`export_gguf.py`

**GGUF 關鍵坑（已解，寫進 export_gguf.py + MODEL_CARD）**：
- Qwen3.5 有 MTP 層 → 預設轉換併主檔致 runtime 報 `missing tensor 'blk.24.attn_norm.weight'`；**主檔須 `--no-mtp`**。
- 工具用官方預編 llama.cpp win 版（b10107 CPU+CUDA），非自編（msys ninja DLL 衝突 0xc0000139 放棄）。
- llama-cli 要 `--chat-template-kwargs '{"enable_thinking":false}'` 關 thinking 才乾淨（否則多空 `<think>`）。

**實測（使用者要求，皆過）**：CPU 跑 ✅ 29 t/s；GPU `-ngl 99` ✅ 128–190 t/s；**MTP 可開** ✅（`--mtp` 匯出 head 當 speculative draft）。皆正確台灣正體、零簡體洩漏。

**待使用者**：實際上傳 HF（需 repo id / hf login）；官方2B+s2twp 出貨組合是否也要打包；雲端 bf16 訓 2B。

## Git 存放庫已初始化（2026-07-26）

`git init -b main` + 首個 commit（74 檔 / 0.45MB，尚未加 remote、未推送）。
- **進版控**：`scripts/ configs/ tools/comet/ docs/ tasks/` + `results/` 的評分 JSON 與 scoreboard.md + 四份 .md + `pyproject.toml`/`uv.lock` + `LICENSE`（Apache-2.0，官方原文下載）
- **gitignore**：`data/ outputs/ release/ logs/ .venv/ results/hyp/`（579 檔 34.8MB 譯文）、`*.gguf`、`.hf_cache/`、`scratchpad/`
- 推送前先建 GitHub repo，再 `git remote add origin <url> && git push -u origin main`

**排除物搶救（2026-07-26）**：gitignore 掉的目錄裡有三樣不可重生／未記錄的證據，已撈進版控＋寫進文件：
- `results/v3/trainer_state.json`、`results/v3-2b/trainer_state.json`（各 ~38KB）——loss 歷史唯一來源，`outputs/` 一刪就永久消失。`plot_loss.py` 預設路徑已改指這裡（原本指 `outputs/`，clone 後跑不動）。
- `docs/assets/loss_curve.png`——GitHub 側曲線（HF 側仍在 `release/assets/`，兩邊都留）。
- `logs/bench/*` 的 8GB 選型數據 + 訓練 wall-clock → 已提煉成 REPORT「訓練成本」表（0.8B 4.01GB/8.0h、2B NF4 5.10GB/17.3h；2B bs2×768=9.11GB 靜默 fallback 270 t/s vs bs1×768=6.12GB 840 t/s）。
- `results/hyp/`（34.8MB）不進版控，但已抽 3 組 base vs v3 譯例寫進 REPORT + MODEL_CARD（全簡體／照抄日文詞 → 修好）。

## 專案目標

把 `Qwen/Qwen3.5-0.8B`（873M，Apache-2.0）LoRA SFT 成 zh-TW↔en↔ja 六方向翻譯特化模型。
計畫全文：`C:\Users\rx595\.claude\plans\qwen-3-5-0-8b-typed-pebble.md`；進度勾選：`tasks/todo.md`。

## 已確認決策

- 本機 3070 Ti 8GB LoRA（不上雲）、純開源語料、純翻譯特化（不混通用資料）
- 起點 Instruct 版；TRL + PEFT；r=32 alpha=64；六方向各 75K 樣本

## 目前進度（2026-07-21 凌晨）

- Phase 0/1/2/3 ✅ 完成。訓練跑完 2660 步 / 2 epochs，耗時 8 小時 27 分，全程零錯誤
- 最終 `train_loss` 2.335、`eval_loss` **2.299**（500 步時為 2.429，一路單調下降無過擬合）
- LoRA adapter：`outputs/sft/adapter_model.safetensors`（86MB）
- Phase 4 評測執行中：`evaluate.py --tag sft-v1 --adapter outputs/sft`，log 在 `logs/eval_sft.log`
- 評測完成後：`uv run --project tools/comet python tools/comet/score.py --tag sft-v1` → 對比 baseline

### eval_loss 走勢

| step | 500 | 1000 | 1500 | 2000 | 2660 |
|---|---|---|---|---|---|
| eval_loss | 2.429 | 2.368 | 2.326 | 2.306 | **2.299** |

### Phase 4 評測結果（sft-v1，最終 checkpoint-2660）

FLORES-200 devtest 500 句/方向。evaluate.py 修了 EOS bug（見 lessons）後的可信數字：

| 方向 | COMET | chrF++ | BLEU | 簡體洩漏 |
|---|---|---|---|---|
| en→zhtw | 86.0→83.8 | 20.9→18.3 | 24.8→23.9 | 20.2%→8.2% |
| zhtw→en | 84.4→84.2 | 48.0→47.5 | 19.4→20.3 | — |
| en→ja | 82.9→81.2 | 18.9→16.1 | 16.9→13.6 | — |
| ja→en | 83.8→84.0 | 45.2→45.6 | 15.9→18.2 | — |
| ja→zhtw | 82.6→83.9 | 11.6→16.0 | 10.3→20.6 | 47.0%→5.8% |
| zhtw→ja | 83.2→81.2 | 14.8→14.4 | 11.8→11.9 | — |
| 平均 | 83.81→83.06 | | | |

**結論是 trade-off，非全勝**：簡體洩漏率暴跌（核心目標達成），但 COMET 平均 -0.75，
6 方向 4 降 2 升。ja→zhtw 唯一三項全贏。生成日文的兩方向（en→ja -1.8、zhtw→ja -2.0）
退步最重，日文資料全來自雜訊多的 OPUS-100 → 疑似後期過擬合 OPUS 風格。

**過訓假設已否證**：COMET 均分 baseline 83.81 > final(ep2.0) 83.06 > ckpt2000(ep1.5) 82.78。
最終 checkpoint 反而優於 ep1.5，6 方向 5 個 final≥ckpt2000 → 沒過訓，第二 epoch 有幫助，
final 為三者最佳。COMET 微降是「翻譯特化 vs 通用能力」的固有 trade-off，非 bug。
（本想測 ckpt1500 但已被 save_total_limit=3 刪除，改用現存最早的 ckpt2000。）

### Phase 4.5 重訓 sft-v2（2026-07-21，最終採用）✅

使用者選了「換乾淨日文語料重訓」。把 en↔ja 從 OPUS-100 換成書面體混配
（WikiMatrix 30K + TED2020 20K + JParaCrawl-filtered 20K + Tatoeba 10K + News-Commentary），
ja↔zhtw 前置一小份 News-Commentary ja-zh。訓練 2978 步 / 2 epoch / ~9.4h，
eval_loss 全程單調下降且低於 v1（2.267→2.165）。**結果全面勝出**：

| 方向 | COMET base→v1→**v2** | 簡體洩漏 base→**v2** |
|---|---|---|
| en→zhtw | 86.0→83.8→**84.63** | 20.2%→**6.4%** |
| zhtw→en | 84.4→84.2→**84.28** | — |
| en→ja | 82.9→81.2→**85.81** ⬆⬆ | — |
| ja→en | 83.8→84.0→**85.08** | — |
| ja→zhtw | 82.6→83.9→**84.56** | 47.0%→**6.4%** |
| zhtw→ja | 83.2→81.2→**83.14** | — |
| **平均** | 83.81→83.06→**84.58** | — |

**v1 的 COMET regression（−0.75）消失並反超 baseline +0.77**；en→ja +2.9 vs baseline（+4.6 vs v1）。
根因是 OPUS-100 日文口語/雜訊 vs FLORES 書面體 domain 不匹配，非「固有 trade-off」（見 lessons）。
唯一小讓步 en→zhtw COMET −1.4（刻意用語意換字形正確，BLEU 24.9 已回 baseline 水準）。

**Phase 5 已 merge v2**：`outputs/merged`（safetensors，generation_config 內建正確 eos_token_id
[248044,248046]）。六方向抽測乾淨無簡體洩漏（`results/merged_check.txt`）。
v1 adapter 保留於 `outputs/sft`，v2 於 `outputs/sft-v2`。

**待使用者決策**：GGUF 發布——用 llama.cpp `convert_hf_to_gguf.py` 轉 outputs/merged 成
Q8_0/Q4_K_M。模型已驗證可用，品質為三版最佳。

### v3 Phase A：多領域診斷（2026-07-22）✅

計畫全文 `C:\Users\rx595\.claude\plans\qwen-3-5-0-8b-typed-pebble.md`。evaluate.py 已泛化為
可插拔 5 基準（flores/ntrex/wmt22/alt/tico19，`--benchmark`）；v2 全面板評測 + COMET 完成
（`results/v2-<bench>.json`、`results/scoreboard.md`）。**方向 × 基準 COMET**（`~`=zhtw 目標
簡體基準 s2twp 參考，次要）：

| 方向 | flores | ntrex(新聞·原生繁) | wmt22(多領域) | alt(Wikinews) | tico19(醫療) |
|---|---|---|---|---|---|
| en→zhtw | 84.6 | **80.4** | 83.8~ | 85.3~ | 85.7~ |
| zhtw→en | 84.3 | 83.0 | **77.8** | 84.6 | 85.8 |
| en→ja | 85.8 | 83.3 | 83.9 | 85.8 | — |
| ja→en | 85.1 | **81.5** | **76.8** | 84.7 | — |
| ja→zhtw | 84.6 | **78.8** | — | 83.8~ | — |
| zhtw→ja | 83.1 | 81.7 | — | 82.8 | — |

**診斷結論**：
1. **確有 FLORES 過擬合，但非全面崩**。最乾淨判據是 NTREX（同為原生繁體、但新聞領域）：六方向
   一致低於 FLORES **−1.3~−5.8**（非參考誤差，是真領域退化）。最重 ja→zhtw −5.8、ja→en −3.6、en→zhtw −4.2。
2. **醫療(TICO)、Wikinews(ALT) 反而穩**（多在 FLORES 水準或更高）→ 過擬合specifically 針對新聞語域，非所有 OOD。
3. **WMT into-English 崩**（ja→en 76.8、zhtw→en 77.8，約 −8）而 out-of-English 守（en→ja 83.9）→
   **源側解析弱**：模型生成英文沒問題，但解析口語/雜訊的 ja/zh 源吃力。
4. **zhtw→ja 全領域一致最低**（83.1/81.7/82.8）→ 非 FLORES 假象，是領域無關的真弱點（呼應 2B 診斷）。
5. **簡體洩漏全領域穩健**（3~8%，OOD 也守住）→ 核心字形修正泛化良好，Phase B 別破壞。

**Phase B 資料方向（據診斷）**：優先補 **新聞語域**（尤其 ja→zhtw，破 TED 壟斷）+ **源側口語/雜訊
的 ja/zh**（救 WMT into-English 崩）；洩漏修正已穩、維持即可。

**待修**：CometKiwi（`Unbabel/wmt22-cometkiwi-da`）① gated 403（HF token 未接受授權）
② 隔離環境 comet 版本 `download_model` 不認得該模型名（KeyError not supported）。非阻塞。

### v3 Phase B：資料多元化（2026-07-22）🔄

- **B1 污染閘 ✅**：`scripts/dump_eval_lines.py` 把 5 基準全量個別語言行（含 s2twp 變體，
  33,984 行）寫 `data/eval_lines.txt`；prepare_data 缺檔硬報錯、任一側命中即丟
  （`eval_contaminated` 統計）。閘在 s2twp 轉換之後比對。
- **B2 ✅**：MAX_SHARE=0.5 防單一來源壟斷；DOMAIN 標籤 + realized 領域計數寫 data_stats.json；
  預算 130k/方向。
- **B3 ✅**（來源皆 workflow 平行查證過）：GlobalVoices en-zhtw 138k / ja-zhtw 18.9k
  （**OPUS 語碼是 jp/zht 不是 ja/zh**；原生繁體新聞；對齊噪 → pre_globalvoices 濾句尾逗號碎片，
  ja-zhtw cap 僅 8k 壓噪；若 v3 ja→zhtw 仍崩再上 LaBSE/COMET-QE）、KDE4 en/ja-zhtw 142k/117k
  （原生台灣繁體 IT；已 tokenized → pre_kde4 detok+濾 %1/&A 佔位符）、KFTT en-ja 440k、
  OpenSub en-ja 2.07M（v2024）、OpenSub en-zhtw 4.77M（v2018）、MTNT ja-en 6.5k（src==tgt 已濾）。
  **ONE_WAY**：opensub.en-ja / mtnt / opensub.en-zhtw 只進 →en 方向源側（救 WMT into-English，
  同時保護 en→ja 產出與 zhtw 洩漏戰果）。GNOME zh 端實質為空（≤78 對）不用；UM/MultiUN 不做。
- **清洗完成**：train 773,389 / dev 1,200（130k/方向）；抽驗洩漏 0.00%（n=5000）。
  領域 realized（見 results/data_stats.json）：ja↔zhtw TED 降至 38% + wiki/subtitle/IT/news；
  en↔zhtw +news 11.3k +IT 12k；→en 方向 +口語源側（opensub/mtnt）。
  **污染答案：v2 來源命中僅 ~0.3%（多為短句通用語碰撞）→ v2 沒實質背考卷，Phase A 診斷結論不變**；
  v3 起污染歸零。未治滿：GV 碎片過濾殺 84%，en↔zhtw news 只到 11k；ja↔zhtw news 僅 3k。
- B4 合成 zhtw→ja **延後**：等 2B 診斷判「資料 vs 能力」再花這筆 GPU。

### v3 Phase C：訓練準備（2026-07-22）🔄

- configs/sft_lora_v3.yaml：0.8B r64/alpha128 + neftune_noise_alpha 5 + 1 epoch → outputs/sft-v3
- configs/sft_qlora_2b.yaml：Qwen3.5-2B NF4 QLoRA，r32/alpha64 沿 v2（診斷可比）→ outputs/sft-2b-qlora
- train_sft.py 加 `quant: nf4` 分支（BitsAndBytesConfig + device_map）與 neftune 接線；
  bench_step.py 加 --model/--r/--nf4。bitsandbytes 0.49.2 已在主環境（Windows wheel OK）
- v2 config（sft_lora.yaml）保留不動當紀錄
- **C0 bench 實測**：0.8B r64 bs2×768 = 7.18GB ✅（r64 只 +0.25GB）；
  2B NF4 bs2×768 = **9.11GB 超標**（靜默 fallback 實錘，270 tok/s）→ 改 **bs1×768 = 6.12GB / 840 tok/s** ✅
  （bs2×512=7.12GB/460 棄）。2B config 已定案 bs1+ga32（有效 batch 32 不變）。
- 預估：0.8B ~10-12h、2B ~17-20h，單 GPU 排隊 ~30h。**等使用者確認才開跑**。

### log 緩衝行為（重要）

`logs/train.log` 在訓練期間只有進度條，**loss 要等 process 結束才 flush**（tqdm/stderr 緩衝）。
中途要看 loss 請讀 `outputs/sft/checkpoint-*/trainer_state.json` 的 `log_history`。

### 硬體調校結論（重要）

3070 Ti 8GB 上記憶體隨 **batch×seq 總 token 數**暴增，主因是 vocab 248K 的 logits 物化。
實測（`scripts/bench_step.py`，皆開 gradient checkpointing）：

| 設定 | VRAM | 吞吐 |
|---|---|---|
| bs2×seq768 | 6.93GB | **1405 tok/s** ← 採用 |
| bs3×seq512 | 6.93GB | 1196 tok/s |
| bs2×seq512 | 5.24GB | 1200 tok/s |
| bs4×seq1024 | 15.3GB（溢出） | 640 tok/s |
| bs8×seq1024 | 28.8GB（溢出） | 280 tok/s |

**Windows 陷阱**：超過 VRAM 不會 OOM，NVIDIA 驅動靜默 fallback 到系統記憶體，
速度掉到 1/5 且無任何警告。第一次全量訓練就是這樣跑了 4 小時才 30 步。
判斷方式：`nvidia-smi` 顯示滿載但吞吐異常低 → 用 bench_step.py 量 VRAM 是否 >8GB。

Liger Kernel 可解 logits 記憶體問題，但只有 Linux wheel，Windows 不可用。

## 舊進度（2026-07-20 下午）

- Phase 0 ✅：環境 OK（torch 2.11+cu128, transformers 5.14.1, trl 1.8.0）；smoke test 過，
  LoRA 10 步峰值 VRAM 3.46GB（bs2×seq512+checkpointing）→ 顯存餘裕大
- Phase 1 🔄：7 份語料下載完（COCT 300K、TED2020 en/ja-zhtw 40萬/36萬、OpenSub 69萬、
  WikiMatrix 27萬、OPUS-100 en-ja/en-zh 各 100萬）；prepare_data.py 清洗執行中
- Phase 2 部分 ✅：evaluate.py 與 COMET 管線已用 8 句 smoke 驗證通過（results/smoketest.json，
  正式 baseline 還沒跑）；基線弱點：zh 目標簡體洩漏 ~37%、ja→zhtw 極弱（chrF++ 6.6）

## 踩過的坑

1. `unbabel-comet` 鎖 transformers<4.58 → qwen3_5 架構載不了；已隔離到 tools/comet 子專案
2. transformers 5.x `apply_chat_template` 回傳 dict：要 `return_dict=True` + `model.generate(**inputs)`
3. 別在 Python 原始碼裡放字面控制字元（null byte 會讓直譯器拒讀）；CTRL_RE 已改 escape 寫法
4. FLORES+ gated 且 API 申請不過 → 改用 Meta 公開 FLORES-200 tarball（dl.fbaipublicfiles.com）
5. Muennighoff/flores200 是 script 型資料集，datasets 3.x 不支援

## 下一步（等使用者拍板）

1. **上傳 HF**：`release/` 已備妥（lora-v3 / merged-bf16 / gguf / MODEL_CARD / loss 曲線）；需 repo id + `hf login`。
2. **打包「官方 Qwen3.5-2B + s2twp」**出貨腳本（零訓練、品質最高，COMET 88.21）——是否也要一起發？
3. **全精度 2B 微調**：租雲端 >8GB GPU 用 bf16 訓一次，才能乾淨判定「2B 微調有無價值」
   （目前 8GB 只能 4-bit QLoRA，量化稅汙染結論）。

> ✅ 已完成（Phase E，2026-07-25）：0.8B v3 打包 + GGUF Q8/Q4 + CPU/GPU 推論實測 + MTP 匯出（見上方 Phase E）。

### 目錄結構（速查）

```
configs/   3 份訓練 config（sft_lora / sft_lora_v3 / sft_qlora_2b）
scripts/   資料管線+訓練+評測（download/prepare/train_sft/evaluate/scoreboard/regression_guard…）
tools/comet/  COMET 隔離子專案（score.py / kiwi.py，獨立 uv 環境，鎖 transformers<4.58）
data/      raw/（21 份 tsv 原始語料）、sft/（train.jsonl 77萬+dev.jsonl）、flores200/、eval_lines.txt（污染閘）
outputs/   sft(v1)/ sft-v2/ sft-v3/ sft-2b-qlora/（LoRA adapter+checkpoint）、merged/（v2 合併全權重）
results/   評測 json 依版本分類：baseline/ v1/ v2/ v3/ v3-2b/（檔名保持 <tag>-<bench>.json，scoreboard.py/regression_guard.py 已改 rglob 遞迴讀）、hyp/（src/ref/hyp 譯文，score.py 硬編 results/hyp/<tag>/ 勿動）、scoreboard.md（總表）、data_stats.json
logs/      執行 log 依類型分：data/ bench/ train/ eval/ comet/ export/（新 log 請寫進對應子目錄）
tasks/     todo.md（進度）、lessons.md（踩坑教訓）
docs/      REPORT.md（研究報告：實驗結果全紀錄）
release/   (gitignored) HF 上傳暫存：lora-v3/ merged-bf16/ gguf/ assets/ + MODEL_CARD.md、.gitattributes
```
