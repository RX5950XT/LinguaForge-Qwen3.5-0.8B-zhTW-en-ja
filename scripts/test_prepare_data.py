"""prepare_data 清洗函數自檢。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import CJK_RE, KANA_RE, eff_len, has_simplified, norm, valid_lang

assert norm("a​b  cd") == "ab cd", repr(norm("a​b  cd"))
assert KANA_RE.search("ひらがな") and KANA_RE.search("カタカナ")
assert not KANA_RE.search("中文")
assert CJK_RE.search("中文") and CJK_RE.search("漢字")
assert valid_lang("Hello world.", "en")
assert not valid_lang("Hello 世界", "en")
assert valid_lang("今日はいい天気です", "ja")
assert not valid_lang("全部漢字", "ja")
assert valid_lang("繁體中文測試", "zhtw")
assert not valid_lang("中文にかな", "zhtw")
assert has_simplified("简体字")
assert not has_simplified("繁體字")
assert eff_len("中文ab") == 6

from prepare_data import is_noise_pair, strip_subtitle_noise

assert strip_subtitle_noise("(ヘレン) 本当にありがとう") == "本当にありがとう"
assert strip_subtitle_noise("HW:：謝謝你。") == "謝謝你。"
assert strip_subtitle_noise("- (拍手) CA: 謝謝") == "謝謝"
assert strip_subtitle_noise("I said: no") == "I said: no"
assert is_noise_pair("♪ la la ♪", "歌詞")
assert is_noise_pair("ありがとう", "謝謝你。謝謝你。")
assert not is_noise_pair("本当にありがとう", "真的很謝謝你。")

# v3 B3 前處理
from prepare_data import detok, pre_globalvoices, pre_kde4, pre_opensub

assert detok("檔案 ( F) 已 儲存 。") == "檔案 (F) 已儲存。", repr(detok("檔案 ( F) 已 儲存 。"))
assert detok("Open file , then save .") == "Open file, then save."
assert pre_kde4("Save %1 now", "立即儲存 %1") is None          # UI 佔位符
assert pre_kde4("&Apply changes", "套用(&A)變更") is None       # 加速鍵
assert pre_kde4("Open file .", "開啟 檔案 。") == ("Open file.", "開啟檔案。")
assert pre_globalvoices("Last Monday,", "上禮拜一，") is None    # 逗號碎片
assert pre_globalvoices("A full sentence.", "完整的句子。") == ("A full sentence.", "完整的句子。")
assert pre_opensub("[grunts] Let's go!", "走吧！") == ("Let's go!", "走吧！")

# B1 污染閘：任一側命中 eval 行即丟；簡體訓練行經 s2twp 後撞 eval 變體也要抓到
import tempfile
from collections import Counter

import prepare_data
from prepare_data import cc_s2twp, load_corpus

with tempfile.TemporaryDirectory() as td:
    old_raw, prepare_data.RAW = prepare_data.RAW, Path(td)
    rows = [
        ("This sentence hits the eval set.", "這句話撞到了評測集喔。"),      # tgt 命中
        ("A perfectly clean training pair.", "一句完全乾淨的訓練資料。"),   # 乾淨
        ("Simplified collides after convert.", "简体转换后才会撞到评测集。"),  # s2twp 後命中
    ]
    (Path(td) / "t.tsv").write_text(
        "".join(f"{a}\t{b}\n" for a, b in rows), encoding="utf-8")
    eval_lines = {"這句話撞到了評測集喔。",
                  cc_s2twp.convert("简体转换后才会撞到评测集。")}
    st = Counter()
    kept = load_corpus("t.tsv", "en", "zhtw", st, eval_lines)
    prepare_data.RAW = old_raw
    assert st["t.tsv:eval_contaminated"] == 2, dict(st)
    assert [b for _, b in kept] == ["一句完全乾淨的訓練資料。"], kept

print("prepare_data helpers OK")
