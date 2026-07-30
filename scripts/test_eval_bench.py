"""eval_bench.py 的題面／答案對映與 logprob 取位自檢（不需 GPU）。

跑：uv run python scripts/test_eval_bench.py
"""

from transformers import AutoTokenizer

from eval_bench import (LETTERS, MODEL_ID, item, letter_ids, mc_prompt,
                        rotate, subsample)


def test_letter_prompt_shape():
    p = mc_prompt("zhtw", "台灣最高峰是？", ["玉山", "雪山", "合歡山", "阿里山"])
    assert "A. 玉山" in p and "D. 阿里山" in p
    assert p.rstrip().endswith("答案："), p[-20:]

    p2 = mc_prompt("en", "Q?", ["a", "b", "c", "d"], passage="CONTEXT HERE")
    assert p2.startswith("CONTEXT HERE\n\n"), p2[:40]


def test_item_keeps_gold_and_options():
    x = item("zhtw", "Q？", ["  甲 ", "乙", "丙", "丁"], 2, passage="P")
    assert x["gold"] == 2
    assert x["options"][0] == "甲", "選項前後空白要去掉，否則會多算一個 token"
    assert len(x["options"]) == 4
    assert "C. 丙" in x["letter"]


def test_rotation_moves_gold_with_options():
    """輪轉算錯的話 acc 會安靜地變成雜訊，一定要獨立驗。"""
    x = item("en", "Q?", ["w", "x", "y", "z"], 2)      # 正解 = "y"
    seen = set()
    for r in range(4):
        v = rotate(x, r)
        opts = [ln.split(". ", 1)[1] for ln in v["letter"].splitlines()
                if len(ln) > 2 and ln[0] in LETTERS and ln[1:3] == ". "]
        assert len(opts) == 4 and set(opts) == {"w", "x", "y", "z"}, opts
        assert opts[v["gold"]] == "y", f"r={r} 正解跑掉了：{opts} gold={v['gold']}"
        seen.add(v["gold"])
    assert seen == {0, 1, 2, 3}, f"正解沒有走完四個位置：{seen}"


def test_gold_index():
    """BELEBELE 的 correct_answer_num 是 1-based 字串；MMLU 系列是字母。"""
    assert int("1") - 1 == 0 and int("4") - 1 == 3
    assert LETTERS.index("C") == 2


def test_subsample_deterministic():
    """同 seed 必須抽到同一批題目，否則跨版本的 Δ 沒有意義。"""
    rows = list(range(500))
    assert subsample(rows, 50, 42) == subsample(rows, 50, 42), "同 seed 抽樣不穩定"
    assert subsample(rows, 50, 42) != subsample(rows, 50, 7), "抽樣沒生效"
    assert len(subsample(rows, 50, 42)) == 50
    assert subsample(rows, 9999, 42) == rows      # limit 大於總數時不可截斷


def test_letters_are_single_token():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = letter_ids(tok)
    assert len(set(ids)) == 4, f"A/B/C/D 撞到同一個 token: {ids}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\n全數通過")
