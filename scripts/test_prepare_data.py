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

# --- v5b：全形標點後的空白（v4/v5a 訓練資料 32.7% 帶著它，輸出放大到 56~76%）---
assert norm("地標景點開始， 我把這裡稱為 「集體記憶」。") == "地標景點開始，我把這裡稱為「集體記憶」。"
assert norm("たとえば、 ある指導者は。 別の指導者は") == "たとえば、ある指導者は。別の指導者は"
assert norm("開始，　我把") == "開始，我把"          # 全形空白 U+3000 也要吃掉
# 拉丁字母／數字兩側的空格是台灣排版慣例，不可誤刪
assert norm("他說， Ring 公司成立") == "他說， Ring 公司成立"
assert norm("我們有 4 個月大的老鼠") == "我們有 4 個月大的老鼠"

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
    # v4: load_corpus 回傳 (lineno, a, b)，行號供文件級重組判斷相鄰性
    assert [b for _, _, b in kept] == ["一句完全乾淨的訓練資料。"], kept
    assert [i for i, _, _ in kept] == [1], kept

# --- v4 噪音過濾 ---
from prepare_data import STRAY_SPACE_RE, punct_asymmetric

assert punct_asymmetric("A full sentence.", "沒有句末標點的譯文")       # 不對稱 → 丟
assert punct_asymmetric("no punctuation here", "有句末標點的譯文。")
assert not punct_asymmetric("A full sentence.", "完整的句子。")          # 兩側都有 → 留
assert not punct_asymmetric("talk transcript style", "講稿風格沒有標點")  # 兩側都無 → 留
assert STRAY_SPACE_RE.search("設定 API_ KEY 環境變數")
assert STRAY_SPACE_RE.search("http: // www. w3. org/")
assert STRAY_SPACE_RE.search("example.com/ zh- tw? ref= news")
assert not STRAY_SPACE_RE.search("設定 API_KEY 環境變數")
assert not STRAY_SPACE_RE.search("Visit https://example.com/zh-tw?ref=news")

# --- v4 文件級重組：只併原檔連號的句子，遇缺號斷開 ---
import random

from prepare_data import DOC_MAX, DOC_MIN, build_docs

rng = random.Random(0)
rows = [(i, f"en{i}.", f"zh{i}。") for i in range(3 * DOC_MAX)]   # 全部連號
docs = build_docs(rows, rng)
assert docs, docs
for a, b in docs:
    assert DOC_MIN <= a.count("\n") + 1 <= DOC_MAX
    ids = [int(x[2:-1]) for x in a.split("\n")]
    assert ids == list(range(ids[0], ids[0] + len(ids))), ids   # 段內必須連號
    assert [int(x[2:-1]) for x in b.split("\n")] == ids         # 兩側同步

gapped = [(i, f"a{i}.", f"b{i}。") for i in (0, 1, 5, 6)]
assert build_docs(gapped, rng) == [], "跨缺號不得併成假文件"     # 兩段各只有 2 句 < DOC_MIN

first = list(range(DOC_MIN))                                     # 剛好夠長的前段
run_then_gap = [(i, f"a{i}.", f"b{i}。") for i in first + [90, 91]]
out = build_docs(run_then_gap, rng)
assert out == [("\n".join(f"a{i}." for i in first),
                "\n".join(f"b{i}。" for i in first))], out       # 只有前段夠長

# 兩側皆無句末標點的行要當斷點（TED ja-zhtw 有 37%），否則整段沒句號
tail = list(range(2, 2 + DOC_MIN))
no_punct = ([(0, "a0.", "b0。"), (1, "a1", "b1")]                 # 索引 1 兩側皆無標點 → 斷點
            + [(i, f"a{i}.", f"b{i}。") for i in tail])
out = build_docs(no_punct, rng)
assert out == [("\n".join(f"a{i}." for i in tail),
                "\n".join(f"b{i}。" for i in tail))], out

# --- v4 標點復原：完整句補句號，碎片原樣退回（交給 punct_asymmetric 丟）---
from prepare_data import restore_punct

assert restore_punct("8年間私はエアフォースツーで飛んでいました", "ja").endswith("。")
assert restore_punct("お礼を申し上げたい", "ja").endswith("。")
assert restore_punct("自分で運転をして", "ja") == "自分で運転をして"        # て＝連用形
assert restore_punct("急にあることに気づきました。", "ja").endswith("た。")  # 已有標點不重複
assert restore_punct("我想感謝大家對我之前演講的好評", "zhtw").endswith("。")
assert restore_punct("全球性變暖污染比", "zhtw") == "全球性變暖污染比"      # 介詞收尾＝碎片
assert restore_punct("雖然", "zhtw") == "雖然"
assert restore_punct("正值晚餐時刻，", "zhtw") == "正值晚餐時刻，"          # 逗號收尾不可補成「，。」

# 補標點要有對側證據；兩側皆無標點＝沒有句界證據，整列丟
from prepare_data import restore_pair

assert restore_pair("A full sentence.", "完整的句子", "en", "zhtw") == \
    ("A full sentence.", "完整的句子。")
assert restore_pair("8年間飛んでいました", "I flew for eight years.", "ja", "en") == \
    ("8年間飛んでいました。", "I flew for eight years.")
assert restore_pair("字幕沒有句號", "字幕也沒有", "ja", "zhtw") is None
assert restore_pair("兩側都有。", "都有標點。", "ja", "zhtw") == ("兩側都有。", "都有標點。")

# CJK 之間的空格是字幕斷行產物，norm 一律清掉（TED 中文側 46% 有）
assert norm("我想感謝大家 對我之前演講的好評。") == "我想感謝大家對我之前演講的好評。"
assert norm("どうもありがとう クリス") == "どうもありがとうクリス"
assert norm("使用 Python 開發") == "使用 Python 開發"                      # 拉丁字旁的空格要留

# --- v4 JSONL 換行陷阱：str.splitlines() 會在 U+2028 等字元斷行，
#     json.dumps 卻不跳脫它們 → 一筆記錄被讀成多行、解析炸掉（bii2155is）
import json

from build_replay import clean

assert " " not in clean("line one line two 這是一段中文", "zhtw")
rec = json.dumps({"t": "a b"}, ensure_ascii=False)
assert len(rec.splitlines()) == 2, "splitlines 會拆壞單行 JSON——讀檔一律用 split('\\n')"
assert len(rec.rstrip("\n").split("\n")) == 1

replay = Path(__file__).parent.parent / "data" / "sft" / "replay.jsonl"
if replay.exists():
    body = replay.read_text(encoding="utf-8").rstrip("\n")
    assert len(body.splitlines()) == len(body.split("\n")), "replay.jsonl 含未跳脫的換行字元"

# --- v4 token 預算組批：每批 padding 後的總 token 不得超過預算（超過就 OOM）
from train_sft import TokenBudgetBatches  # noqa: E402

lens = [7, 300, 1, 768, 88, 88, 500, 12, 12, 12, 400]
bs = TokenBudgetBatches(lens, budget=1450, seed=0)
seen = sorted(i for b in bs.batches for i in b)
assert seen == list(range(len(lens))), "每筆樣本要且只要出現一次"
for b in bs.batches:
    assert len(b) * max(lens[i] for i in b) <= 1450, f"批次超出 token 預算: {b}"
assert list(iter(bs)) and len(list(iter(bs))) == len(bs)

# --- v5 注水式來源配額：v4 的依序貪婪在小預算下只餵得到前 2 個來源，
#     每方向的領域組合整個消失（見 docs/RESEARCH-v5.md F1）
from prepare_data import waterfill  # noqa: E402

# v4 實況重現：sent_budget 17,000、hard 8,500、五個來源。
# 舊寫法給出 [8500, 8500, 0, 0, 0]；注水式要讓五個來源都拿到額度。
takes = waterfill([8500] * 5, 17_000)
assert sum(takes) == 17_000 and all(t > 0 for t in takes), takes

# 小池子取不滿的餘額要流向大池子，不能讓總額短收
takes = waterfill([12_000, 15_000, 21_006, 50_000, 55_250], 110_500)
assert sum(takes) == 110_500, takes
assert takes[:3] == [12_000, 15_000, 21_006], "上限低於均分額的來源應該全取"

# 預算超過所有上限總和 → 每個來源取滿即止，不得超取
assert waterfill([5, 10], 999) == [5, 10]
assert waterfill([], 100) == [] and waterfill([3, 3], 0) == [0, 0]


# --- v5b LaBSE 語意過濾（F7）：三條分支都要擋得住 -----------------------------
# 這段是硬報錯路徑，出錯時機在「LaBSE 掃完 40 分鐘之後」，沒有自檢會很貴。
import numpy as np  # noqa: E402
from collections import Counter  # noqa: E402

from prepare_data import ROOT, labse_filter  # noqa: E402

_rows = [(0, "a", "甲"), (1, "b", "乙"), (2, "c", "丙"), (3, "d", "丁")]
_cache = ROOT / "data" / "labse" / "__selftest__.tsv.npz"
_cache.parent.mkdir(parents=True, exist_ok=True)

# min_score = 0 → 直接放行，連快取都不該碰（--labse-min 0 的逃生口）
assert labse_filter("__never_exists__.tsv", _rows, 0, Counter()) is _rows

np.savez(_cache, lineno=np.array([0, 1, 2, 3]),
         score=np.array([0.95, 0.55, 0.60, 0.10], dtype=np.float32))
st = Counter()
kept = labse_filter("__selftest__.tsv", _rows, 0.60, st)
assert [r[0] for r in kept] == [0, 2], kept          # 0.60 是「大於等於」，邊界要留
assert st["__selftest__.tsv:labse_dropped"] == 2

# 快取涵蓋率不足 → 必須硬報錯，不可拿舊分數矇混（清洗規則一改行號就對不上）
np.savez(_cache, lineno=np.array([0]), score=np.array([0.9], dtype=np.float32))
try:
    labse_filter("__selftest__.tsv", _rows, 0.60, Counter())
    raise AssertionError("涵蓋率 25% 應該要 SystemExit")
except SystemExit as e:
    assert "不同步" in str(e), e

# 快取缺檔 → 硬報錯（語意過濾為強制，要關掉必須明示 --labse-min 0）
_cache.unlink()
try:
    labse_filter("__selftest__.tsv", _rows, 0.60, Counter())
    raise AssertionError("缺快取應該要 SystemExit")
except SystemExit as e:
    assert "bitext_filter.py" in str(e), e

print("prepare_data helpers OK")
