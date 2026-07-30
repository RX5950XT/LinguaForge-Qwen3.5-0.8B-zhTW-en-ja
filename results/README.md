# results/ 目錄慣例

`evaluate.py` 一律把新結果寫在**根目錄** `results/<tag>-<benchmark>.json`。
評測告一段落後，手動歸檔進對應版本子目錄即可——所有讀取端都用
`rglob` 定位（`scoreboard.py`、`regression_guard.py`、`tools/comet/{score,kiwi}.py`），
分類後照樣找得到，不必改程式。

```
baseline/   base 模型各種設定（greedy / beam4 / int8 / nf4 / 2B）
v1/ v2/ v3/ v3-2b/ v4/ v5/    各版翻譯評測（FLORES / NTREX / WMT22 / ALT / TICO-19）
capability/ 能力面板（eval_capability.py 直接寫這裡，路徑固定，不要再往下分層）
bench/      公開知識／常識基準（eval_bench.py 直接寫這裡，同上不要分層）
hyp/        逐句 src/ref/hyp 文字檔（.gitignore 排除，COMET 靠它重算）
```

`bench/` 檔名帶計分法：`<tag>.json` 是輪轉去偏的正式數字，`<tag>-letter.json`
是單輪診斷用。**兩者分數不可互相比較**，見 `eval_bench.py` 開頭說明。

根目錄常駐、**不要移動**（腳本寫死路徑）：

| 檔案 | 寫入者 |
|---|---|
| `data_stats.json` | `prepare_data.py` |
| `merged_check.txt` | `verify_merged.py` |
| `v3/trainer_state.json` | `plot_loss.py` 的 `DEFAULT_STATE` |
| `scoreboard.md` | 手動存的 `scoreboard.py` 輸出 |
