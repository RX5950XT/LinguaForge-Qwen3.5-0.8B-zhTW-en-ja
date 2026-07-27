# RESEARCH-v5 — v4 事後診斷與 v5 設計

v4 訓練完成後的資料稽核結果。**結論先講：v4 最大的問題不是「翻譯資料量砍太兇」，
而是 `--limit` 和 `MAX_SHARE` 交互作用把六個方向的語料配方悄悄截斷成只剩前 2 個來源。**

---

## F1（主因）`--limit 20000` 讓每個方向只吃到 2 個語料來源

`scripts/prepare_data.py` 的 `RECIPES` 每方向預算是 **130,000**（從未改過），
v4 是用 `--limit 20000` 跑的——而 `--limit` 在檔頭註解裡寫的是「10% 子集驗證用」。

配額計算：

```python
budget      = min(130_000, 20_000) = 20_000
doc_budget  = 20_000 * 0.15        =  3_000
sent_budget = 17_000
hard        = 17_000 * MAX_SHARE(0.5) = 8_500   # 防單一來源壟斷
```

來源迴圈是**依序貪婪**取 `min(room, cap, hard, len(pool))`，於是：

| 方向 | 實際取到的來源（data_stats.json 實測） | 領域 | wiki? |
|---|---|---|---|
| en→zhtw | coct 8,500 + globalvoices 8,500 | textbook, news | ❌ |
| zhtw→en | coct 8,500 + globalvoices 8,500 | textbook, news | ❌ |
| en→ja | wikimatrix 8,500 + jparacrawl 8,500 | wiki, web | ✅ |
| ja→en | wikimatrix 8,500 + jparacrawl 8,500 | wiki, web | ✅ |
| ja→zhtw | newscomm 258 + globalvoices 2,653 + ted2020 8,500 + wikimatrix 5,589 | news, talk, wiki | ✅ |
| zhtw→ja | 同上 | news, talk, wiki | ✅ |

**ted2020 / kde4 / opus100 / kftt / tatoeba / opensub 在 en↔zhtw 與 en↔ja 完全沒被取到**
（只有 ja↔zhtw 因為前幾個池子小才輪得到第 3、4 個來源）。
`MAX_SHARE` 本來是防壟斷，在小預算下反而變成「兩個來源就把 room 吃光」。

### 這解釋了 COMET 的分佈

| 方向 | base | v3 | v4 | v4−v3 |
|---|---|---|---|---|
| en→zhtw | 85.92 | 85.08 | **82.94** | **−2.14** |
| ja→zhtw | 82.69 | 84.74 | 83.49 | −1.25 |
| zhtw→en | 84.40 | 84.47 | 83.84 | −0.63 |
| zhtw→ja | 83.07 | 84.09 | 83.48 | −0.61 |
| en→ja | 83.20 | 85.55 | 84.98 | −0.57 |
| ja→en | 83.82 | 84.89 | 84.91 | +0.02 |
| 平均 | 83.85 | 84.80 | 83.94 | −0.86 |

**關鍵反證：en→ja 跟 en→zhtw 被砍掉的資料量完全一樣（130k→20k），
但 en→ja 只掉 0.57、還比 base 高 1.78，en→zhtw 掉 2.14、比 base 低 2.98。**
資料量若是主因，兩者應該一起掉。所以：

- 「130k→20k ＋ 加 replay」的共同代價 ≈ **0.6 COMET**（六方向裡五個都落在這個帶）
- en→zhtw 多掉的 ~1.5 分，來自它是唯一「zh-TW 輸出 ＋ 沒有 wiki 領域」的組合
- 兩個 zh-TW 輸出方向（en→zhtw −2.14、ja→zhtw −1.25）就是掉最多的兩個，
  目標側語域（textbook/news vs wiki）對 zh-TW 產出的影響遠大於對英文產出

**這修正了先前「翻譯資料砍太兇、v5 要拉到每方向 60,000」的判斷。**
量不是主因，配方截斷才是；而修配方幾乎不用多花 GPU。

### 修法與驗證（已實作）

`prepare_data.py` 新增 `waterfill(ceilings, budget)`：由上限小的來源開始，
每輪把剩餘預算平均分給尚未分配者，取不滿的餘額自動流向大池子。
句級與文件級兩個迴圈共用（文件級本來就是這個寫法，等於把正確的那個抄到錯的那邊）。
自檢在 `scripts/test_prepare_data.py`。

用 `data_stats.json` 的真實 `kept` 數量模擬前後領域數：

| 方向 | 舊（budget 20,000） | 新（budget 20,000） | 舊（130,000） | 新（130,000） |
|---|---|---|---|---|
| en→zhtw | 2 域 | **5 域** | 5 域 | 5 域 |
| zhtw→en | 2 域 | **6 域** | 6 域 | 6 域 |
| en→ja | 2 域 | **5 域** | 3 域 | **5 域** |
| ja→en | 2 域 | **7 域** | 4 域 | **7 域** |
| ja→zhtw | 3 域 | **5 域** | 5 域 | 5 域 |
| zhtw→ja | 3 域 | **5 域** | 5 域 | 5 域 |

順帶發現：**即使在 130,000 的完整預算下，舊寫法也讓 en→ja 只吃到 3 域、ja→en 只吃到 4 域**
（`wikimatrix` cap 35,000 加 `jparacrawl` 30,000 加 `kftt` 30,000 就快把 110,500 吃完，
`tatoeba` / `newscomm` / `mtnt` 幾乎輪不到）。v3 本身也沒拿到設計中的配方。

### 語料存量完全不是瓶頸

清洗後 `kept` 總計約 **7.5M** 對，v4 只用了 120,000（1.6%）。單看沒被碰過的：
opensub.en-zhtw 1,035,690、opus100 612,916、opensub.en-ja 550,483、
kftt 333,394、opensub.ja-zhtw 430,423、tatoeba 214,370。

---

## F2 replay 的 22.85% 是「池子見底」，不是設計

`REPLAY_SHARE = 0.35`，反推需要 63,969 筆，但 `replay.jsonl` 只有 **35,177** 筆，
`load_replay` 印了警告後照樣輸出 → 實際 22.85%。先前把它當成「貼近 Tower+ 的 22%」是巧合。

語言分布（以假名/漢字偵測實測）：

| ja | en | zh |
|---|---|---|
| 17,170 | 16,554 | **1,453** |

**這是 v5 擴量的硬牆**：若翻譯樣本放大 3 倍而 replay 池不動，
replay 佔比會從 22.85% 掉到 ~9%，v3 的災難性遺忘就會回來。

### 授權可用的中文 replay 來源——目前確認無解

| 資料集 | 授權 | 判定 |
|---|---|---|
| `BAAI/Infinity-Instruct` | CC-BY-SA-4.0＋gated | ❌ 傳染性 share-alike，且拿不到 |
| `Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese` | **未宣告授權** | ❌ 同 COIG-CQIA 的問題 |
| `Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1` | **未宣告授權** | ❌ |
| `yentinglin/TaiwanChat`、`lianghsun/tw-instruct-500k` | CC-BY-NC | ❌ 與 Apache-2.0 衝突 |
| `aya_collection` traditional_chinese | Apache-2.0 | ❌ 機翻 Flan，序號被當文字翻譯（已於 build_replay.py 記錄） |

→ 中文 replay 要擴，**只剩 Qwen3.5-2B 自蒸餾**（F12，實測 18 組/分鐘，6K 約 5.5 GPU 小時）。
但見 F4：目前沒有證據顯示中文 replay 真的不夠。

---

## F3 文件級訓練樣本比評測文件短 3～6 倍

`DOC_MIN, DOC_MAX = 3, 6`。實測 v4 `train.jsonl` token 長度（Qwen3.5 tokenizer）：

| | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 全部翻譯樣本 | 78 | 183 | 338 | **581** |
| 文件級樣本 | 202 | 315 | 502 | **581** |

`max_length: 768` 一筆都沒濾掉——**它根本不是限制，`DOC_MAX = 6` 才是。**

對照評測端：WMT24++ en-zh_TW 共 171 篇、59 篇 ≥3 句，句數 p50 = 11、p90 = 28、max = 76
（軸 A 取的 12 篇平均 19.2 句）。

**模型訓練時從沒看過超過 581 token 的序列，評測時卻要它一口氣生成到 2048 token。**
日文側 75~100% 的貪婪迴圈，最可能就是這個長度外插造成的（`--rep-penalty 1.1` 實驗正在驗證
「是解碼問題還是能力問題」，但無論結果如何，訓練長度該補）。

放長的成本是零：`configs/sft_v4.yaml` 已實測 `bs1×1450 = 6.90GB / 1300~1600 tok/s`，
token 預算組批本來就是動態 batch size，`max_length` 直接拉到 1408 不需要改任何記憶體策略。

---

## F4 評測面板小到不能拿來做 v5 決策

| 軸 | n |
|---|---|
| general | **12** |
| ifeval | **17**（zhtw 6 / ja 5 / en 6） |
| doc（軸 A） | 12 篇 × 6 方向 |
| FLORES（COMET/chrF++） | 1,012 句 × 6 方向 ✅ |

先前報告的「zhtw ifeval 從 base 66.7 退到 50.0，是 v4 唯一的退步」——
**那是 4/6 → 3/6，差一題。** 不足以支撐任何結論，更不足以支撐花 5.5 GPU 小時去補中文 replay。

同理 general 83.3% = 10/12。只有 FLORES/COMET 那組數字有統計意義。

軸 A 的重複率（en→ja 9/12 = 75%）樣本雖小但效應量夠大，仍可信。

**WMT24++ 文件池還有 59 篇可用（v4 只用 12～25 篇），擴軸 A 只要多花 GPU 時間，不用找資料。**

---

## v5 計畫

順序刻意排成「先把便宜且會影響判讀的東西修掉，再決定要不要花大錢」。

### 階段 0 — 不用 GPU

1. **修 `prepare_data.py` 的來源配額**：把依序貪婪改成注水式分配
   （檔案裡的文件級迴圈 `share = (doc_budget - len(docs)) // (len(avail) - i)` 已經是正確寫法，
   照抄到句級迴圈即可），`cap` 與 `MAX_SHARE` 退回純上限角色。
   → 任何預算下都拿得到完整領域組合。
2. **`--limit` 加防呆**：這個旗標的註解寫「10% 子集驗證用」，卻被用在正式訓練上。
   要嘛印警告，要嘛把正式預算寫進 `RECIPES` 而不是靠 CLI 覆寫。
3. **擴評測面板**：`IFEVAL` / `GENERAL` 是 `eval_capability.py` 裡手寫的清單，
   擴到每語言 ~50 題純粹是撰寫工作，沒有授權問題也不用 GPU。
4. `DOC_MIN, DOC_MAX = 3, 6` → **`4, 16`**；`max_length: 768` → **1408**。
   （16 句 × ~45 tok ≈ 720 tok/側，尾巴超長的少數樣本會被 train_sft 的過濾器丟掉，可接受。）

### 階段 1 — v5a：只動配方，不動量（約 9.5 小時）

維持每方向 20,000、replay 池不變，驗證 F1 的因果。

預估 token 量：句級 17,000×6×~105 = 10.7M ＋ 文件級 3,000×6×~585 = 10.5M
＋ replay 35,177×~203 = 7.1M ≈ **28.3M/epoch**，2 epoch 56.6M ÷ 1,652 tok/s ≈ **9.5 小時**
（v4 實測 42.7M / 25,850s = 1,652 tok/s 端到端）。replay 佔比回到 ~25%。

判讀規則：
- **en→zhtw COMET 回到 85 以上** → F1 成立，量從來不是問題，v5a 直接發布，省下 21.6 小時的擴量跑。
- 回不去 → 才進階段 2 擴量，此時 replay 池必須同步擴（F2），否則會退回 v3 的遺忘問題。
- 軸 A 日文側重複率若隨 `DOC_MAX` 放長而下降 → F3 成立。

### 階段 2 — 只在階段 1 失敗時才做

擴到每方向 60,000（總 ~46 萬筆、~21.6 小時），並先跑 F12 的 2B 自蒸餾把中文 replay
從 1,453 補到 ~8,000，維持 replay ≥ 22% 的 token 佔比。

### 不做

- 不再找中文指令資料集（F2 已窮舉，授權全部不合格）
- 不動 LoRA 超參（r64/alpha128、lr 1e-4、2 epoch 的 eval_loss 在 epoch 2.0 仍在最低點，沒有過擬合跡象）
- 不編 `causal_conv1d`（見 CLAUDE.md：PyPI 無 wheel，要補 CUDA 12.8 toolkit，收益只有解碼時幾個 kernel launch）
