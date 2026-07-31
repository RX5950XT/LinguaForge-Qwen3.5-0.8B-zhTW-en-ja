"""eval_gguf 輸出解析自檢（不需 GPU、不跑 llama-cli）。

寫這支的原因：把 `body[-1]`（猜最後一行是譯文）改成按行數精準切時，
少算了「`> ` 那行本身就是提示詞第一行」，100 句全變空字串才被逐行比對抓到。
版面規則只有一條，但踩過一次就值得夾住。

  uv run python scripts/test_eval_gguf.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_gguf import parse_output  # noqa: E402

# 真實版面：`> ` 那行 ＝ 提示詞第 1 行，接著提示詞第 2 行，然後才是輸出
REAL = """build      : b9437-aa46bda89
using custom system prompt

> 翻譯成繁體中文：
The patient should take this medication twice a day.

病人應每天服用此藥兩次。

[ Prompt: 328.3 t/s | Generation: 139.8 t/s ]
"""
assert parse_output(REAL, 2) == "病人應每天服用此藥兩次。"

# 多行譯文：舊的 body[-1] 會只留最後一行，這裡必須全留
MULTI = REAL.replace("病人應每天服用此藥兩次。", "第一行。\n第二行。")
assert parse_output(MULTI, 2) == "第一行。 第二行。"

# 模型完全沒吐東西 → 空字串，不得把提示詞當成譯文回傳
EMPTY = REAL.replace("病人應每天服用此藥兩次。\n", "")
assert parse_output(EMPTY, 2) == ""

# 沒有 `> ` 或沒有統計行（llama-cli 掛掉）→ 空字串，不得丟例外
assert parse_output("模型載入失敗", 2) == ""
assert parse_output("> 翻譯成繁體中文：\nabc\n譯文", 2) == ""

print("eval_gguf.parse_output OK — 單行/多行/空輸出/壞輸出四種版面皆正確")
