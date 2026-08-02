"""bench_defects 缺陷判定自檢（不需 GPU）：uv run python scripts/test_bench_defects.py"""

from bench_defects import CASES, defects, find_repetition_loop, find_tag_prefix


def test_tag_prefix():
    # 原文沒有標籤 → 判污染
    assert find_tag_prefix("Battery life: 18 hours", "說明：電池壽命：18 小時") == "label"
    assert find_tag_prefix("What are your predictions?", "問：您的預測是什麼？") == "label"
    assert find_tag_prefix("Revenue grew 47%.", "3. 收入增長 47%。") == "enum"
    assert find_tag_prefix("Open weight release", "選擇重量釋放") == "select"
    assert find_tag_prefix("A pig would break through.", "故事說，豬會撞破。") == "narrate"
    assert find_tag_prefix("Jean Guerrero", "圖：Jean Guerrero") == "figure"
    # 乾淨譯文 → 不判
    assert find_tag_prefix("Battery life: 18 hours", "電池壽命：18 小時") is None
    # 原文本來就有同樣標籤 → 不可誤傷
    assert find_tag_prefix("Note: keep it dry", "註：保持乾燥") is None
    assert find_tag_prefix("1. First step", "1. 第一步") is None


def test_repetition_loop():
    assert find_repetition_loop("好的好的好的好的好的好的好的好的") is not None
    assert find_repetition_loop("我預測它會超越 Kimi k3，並可能接近 Sol。") is None
    assert find_repetition_loop("短") is None       # < 16 字不判


def test_defects():
    case = ("t1", "The NVIDIA H200 has 141GB of memory.", "zhtw", ["NVIDIA", "H200"])
    assert defects(case, "NVIDIA H200 擁有 141GB 記憶體。") == []
    assert "keep:NVIDIA" in defects(case, "141GB 記憶體。")
    # C 類：原文沒有的年份
    yr = ("t2", "TSMC will start 2nm production next year.", "zhtw", [])
    assert any(d.startswith("year:2009") for d in defects(yr, "2009 年，TSMC 將開始 2nm 生產。"))
    assert not any(d.startswith("year") for d in defects(yr, "TSMC 明年將開始 2nm 生產。"))
    # 原文有的年份原樣譯出 → 不判
    keep_yr = ("t3", "Released on November 24, 2025.", "zhtw", [])
    assert not any(d.startswith("year") for d in defects(keep_yr, "於 2025 年 11 月 24 日發布。"))
    # D 類：多行壓成單行
    ml = ("t4", "a\nb\nc", "zhtw", [])
    assert "lines" in defects(ml, "甲")
    assert "lines" not in defects(ml, "甲\n乙\n丙")
    assert defects(ml, "") == ["empty"]


def test_cases_intact():
    assert len(CASES) == 30
    assert {c[2] for c in CASES} == {"zhtw", "en", "ja"}
    assert sum(len(c[3]) for c in CASES) == 24   # keep 詞總數，改樣本時要一起改門檻


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("all passed")
