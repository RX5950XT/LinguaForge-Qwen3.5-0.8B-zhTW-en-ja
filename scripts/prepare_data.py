"""Phase 1: 清洗 data/raw/*.tsv → 六方向 SFT JSONL (data/sft/train.jsonl, dev.jsonl)。

流程：控制字元清理 → 去重 → 長度/比例過濾 → 語言驗證 → OpenCC 統一台灣正體
→ 污染閘（任一側命中 eval 基準行即丟）→ 六方向配比抽樣 → chat messages JSONL。
每步統計寫 results/data_stats.json。

污染閘（B1，強制）：data/eval_lines.txt 由 scripts/dump_eval_lines.py 產生，
含全部 eval 基準的個別語言行（含 s2twp 變體）。缺檔直接報錯，不可跳過。

用法：
  uv run python scripts/prepare_data.py            # 全量
  uv run python scripts/prepare_data.py --limit 7500   # 每方向上限（10% 子集驗證用）
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).parent.parent
RAW, SFT = ROOT / "data" / "raw", ROOT / "data" / "sft"
SEED = 42
DEV_PER_DIR = 200

# (file, l1, l2)   l2 一律是 zh 端（若有）
CORPORA = [
    ("coct.en-zhtw.tsv", "en", "zhtw"),
    ("ted2020.en-zhtw.tsv", "en", "zhtw"),
    ("globalvoices.en-zhtw.tsv", "en", "zhtw"),
    ("kde4.en-zhtw.tsv", "en", "zhtw"),
    ("opensub.en-zhtw.tsv", "en", "zhtw"),
    ("opus100.en-zh.tsv", "en", "zhtw"),
    ("ted2020.ja-zhtw.tsv", "ja", "zhtw"),
    ("opensub.ja-zhtw.tsv", "ja", "zhtw"),
    ("wikimatrix.ja-zh.tsv", "ja", "zhtw"),
    ("newscomm.ja-zh.tsv", "ja", "zhtw"),
    ("globalvoices.ja-zhtw.tsv", "ja", "zhtw"),
    ("kde4.ja-zhtw.tsv", "ja", "zhtw"),
    # 乾淨書面體 en-ja（取代 opus100.en-ja）
    ("wikimatrix.en-ja.tsv", "en", "ja"),
    ("ted2020.en-ja.tsv", "en", "ja"),
    ("jparacrawl.en-ja.tsv", "en", "ja"),
    ("tatoeba.en-ja.tsv", "en", "ja"),
    ("newscomm.en-ja.tsv", "en", "ja"),
    ("kftt.en-ja.tsv", "en", "ja"),
    ("opensub.en-ja.tsv", "en", "ja"),
    ("mtnt.ja-en.tsv", "en", "ja"),
]

# 口語/噪聲來源只進 →en 方向的「源側」：保護 en→ja 產出品質與 zhtw 洩漏戰果，
# 同時直接救診斷出的 WMT into-English 崩（源側解析弱）
ONE_WAY = {
    "opensub.en-ja.tsv": ("ja", "en"),
    "mtnt.ja-en.tsv": ("ja", "en"),
    "opensub.en-zhtw.tsv": ("zhtw", "en"),
}

# 每語料領域標籤：realized 計數寫進 data_stats.json（B2 可觀測性）
DOMAIN = {
    "coct.en-zhtw.tsv": "textbook", "ted2020.en-zhtw.tsv": "talk",
    "globalvoices.en-zhtw.tsv": "news", "kde4.en-zhtw.tsv": "it",
    "opensub.en-zhtw.tsv": "subtitle", "opus100.en-zh.tsv": "web",
    "ted2020.ja-zhtw.tsv": "talk", "opensub.ja-zhtw.tsv": "subtitle",
    "wikimatrix.ja-zh.tsv": "wiki", "newscomm.ja-zh.tsv": "news",
    "globalvoices.ja-zhtw.tsv": "news", "kde4.ja-zhtw.tsv": "it",
    "wikimatrix.en-ja.tsv": "wiki", "ted2020.en-ja.tsv": "talk",
    "jparacrawl.en-ja.tsv": "web", "tatoeba.en-ja.tsv": "short",
    "newscomm.en-ja.tsv": "news", "kftt.en-ja.tsv": "wiki",
    "opensub.en-ja.tsv": "subtitle", "mtnt.ja-en.tsv": "ugc",
}

# 每方向：依序從各語料抽 cap 筆（None=不設上限），總額 budget；
# MAX_SHARE 防單一來源壟斷（v2 診斷：TED 壟斷 ja↔zhtw → NTREX 新聞域 -5.8）
MAX_SHARE = 0.5

EN_ZHTW = [("coct.en-zhtw.tsv", 50_000), ("globalvoices.en-zhtw.tsv", 35_000),
           ("ted2020.en-zhtw.tsv", 15_000), ("kde4.en-zhtw.tsv", 12_000),
           ("opus100.en-zh.tsv", None)]
ZHTW_EN = [("coct.en-zhtw.tsv", 50_000), ("globalvoices.en-zhtw.tsv", 35_000),
           ("ted2020.en-zhtw.tsv", 15_000), ("kde4.en-zhtw.tsv", 12_000),
           ("opensub.en-zhtw.tsv", 15_000), ("opus100.en-zh.tsv", None)]
EN_JA = [("wikimatrix.en-ja.tsv", 35_000), ("jparacrawl.en-ja.tsv", 30_000),
         ("kftt.en-ja.tsv", 30_000), ("ted2020.en-ja.tsv", 20_000),
         ("tatoeba.en-ja.tsv", 10_000), ("newscomm.en-ja.tsv", None)]
JA_EN = [("wikimatrix.en-ja.tsv", 30_000), ("jparacrawl.en-ja.tsv", 25_000),
         ("kftt.en-ja.tsv", 25_000), ("ted2020.en-ja.tsv", 15_000),
         ("opensub.en-ja.tsv", 20_000), ("mtnt.ja-en.tsv", None),
         ("tatoeba.en-ja.tsv", 8_000), ("newscomm.en-ja.tsv", None)]
JA_ZHTW = [("newscomm.ja-zh.tsv", None), ("globalvoices.ja-zhtw.tsv", 8_000),
           ("ted2020.ja-zhtw.tsv", 50_000), ("wikimatrix.ja-zh.tsv", 30_000),
           ("kde4.ja-zhtw.tsv", 12_000), ("opensub.ja-zhtw.tsv", None)]
RECIPES = {
    ("en", "zhtw"): (130_000, EN_ZHTW),
    ("zhtw", "en"): (130_000, ZHTW_EN),
    ("en", "ja"): (130_000, EN_JA),
    ("ja", "en"): (130_000, JA_EN),
    ("ja", "zhtw"): (130_000, JA_ZHTW),
    ("zhtw", "ja"): (130_000, JA_ZHTW),
}

TEMPLATES = {
    "zhtw": ["翻譯成繁體中文：", "翻譯成臺灣正體中文：", "Translate to Traditional Chinese (Taiwan):", "台湾の繁体字中国語に翻訳してください："],
    "en": ["翻譯成英文：", "Translate to English:", "英語に翻訳してください："],
    "ja": ["翻譯成日文：", "Translate to Japanese:", "日本語に翻訳してください："],
}
SYSTEM = "You are a professional translator."

# --- v4：文件級樣本 -----------------------------------------------------------
# 只有「逐行連貫且對齊正確」的語料能重組成段落。已逐檔抽樣目檢：
#   ted2020    ✅ 同一場演講連續逐句
#   newscomm   ✅ 同一篇評論連續逐句
#   globalvoices ❌ 中文側切分較細導致累積偏移（ZH[i] 對到 EN[i+1]）
#   wikimatrix / coct / kftt ❌ 挖掘或獨立句，相鄰兩行無關
DOC_SOURCES = {"ted2020.en-zhtw.tsv", "ted2020.ja-zhtw.tsv", "ted2020.en-ja.tsv",
               "newscomm.ja-zh.tsv", "newscomm.en-ja.tsv"}
# v4 用 3, 6 → 訓練樣本最長只有 581 token，但軸 A 的 WMT24++ 文件中位數 11 句、
# p90 28 句，等於要模型外插 3~6 倍長度，日文側 75~100% 陷入貪婪迴圈。
# 16 句 × ~45 tok ≈ 720 tok/側，配 max_length 1408 放得下（bs1×1450 實測 6.90GB）。
DOC_MIN, DOC_MAX = 4, 16     # 每個文件級樣本併幾句
DOC_SHARE = 0.15             # 每方向預算裡文件級樣本的佔比

# --- v4：通用 replay ---------------------------------------------------------
# 100% 翻譯任務 → 模型把所有 user 輸入都當待翻譯文本（災難性遺忘，見 docs/RESEARCH-v4.md）
REPLAY_FILE = SFT / "replay.jsonl"
REPLAY_SHARE = 0.35          # replay 佔最終 train 的比例（Tower+ SFT 用 78%，此處保守）

CTRL_RE = re.compile("[\u0000-\u0008\u000b-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
KANA_RE = re.compile("[぀-ヿ]")
CJK_RE = re.compile("[一-鿿㐀-䶿]")

cc_s2twp = OpenCC("s2twp")
# 洩漏偵測用 s2tw 不用 s2t：s2t 會把「剛才→剛纔」「人群→人羣」「稽核→稽覈」
# 這些正確的台灣用字判成簡體（傳統異體字，非簡體），造成大量偽陽性。
cc_s2tw = OpenCC("s2tw")


SPEAKER_PAREN_RE = re.compile(r"^\s*[（(][^（）()]{1,25}[）)]\s*")
SPEAKER_TAG_RE = re.compile(r"^\s*[A-Z][A-Za-z]{0,10}\s*[:：]{1,2}\s*")
DASH_RE = re.compile(r"^\s*[-－—]+\s*")
MUSIC_RE = re.compile(r"[♪♫♬]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = CTRL_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    # 中日文字之間的空格一律是字幕斷行／分詞產物（中日文正常不用空格分隔），
    # 留著會教模型在中文輸出裡亂插空格：TED2020 中文側 46% 的行有這種空格。
    return CJK_GAP_RE.sub("", s)


def strip_subtitle_noise(s: str) -> str:
    """去掉字幕講者標記：(ヘレン)、HW:、開頭破折號（最多剝三層）。"""
    for _ in range(3):
        t = DASH_RE.sub("", SPEAKER_TAG_RE.sub("", SPEAKER_PAREN_RE.sub("", s)))
        if t == s:
            break
        s = t
    return s.strip()


# --- v3 Phase B3 per-file 前處理（回傳 (a, b) 或 None=丟棄）---
PLACEHOLDER_RE = re.compile(r"%\d|&[A-Za-z]\b|@[a-z]\w+|\$\{|</?\w[^>]*>")
SPACE_BEFORE_RE = re.compile(r"\s+([,.;:!?%)\]、。，；：！？」』】）])")
SPACE_AFTER_RE = re.compile(r"([(\[「『【（])\s+")
CJK_GAP_RE = re.compile(r"(?<=[一-鿿㐀-䶿぀-ヿ])\s+(?=[一-鿿㐀-䶿぀-ヿ])")
BRACKET_NOISE_RE = re.compile(r"[\[(][^)\]]{1,30}[)\]]")
FRAGMENT_END_RE = re.compile(r"[,，、;；:：]$")


def detok(s: str) -> str:
    """KDE4 文字已 tokenized（標點旁多餘空格）→ 還原。"""
    s = SPACE_BEFORE_RE.sub(r"\1", s)
    s = SPACE_AFTER_RE.sub(r"\1", s)
    return CJK_GAP_RE.sub("", s)


def pre_kde4(a: str, b: str):
    """UI 佔位符 %1、加速鍵 &A、context 標記、殘留 markup → 整列丟；其餘 detokenize。"""
    if PLACEHOLDER_RE.search(a) or PLACEHOLDER_RE.search(b):
        return None
    return detok(a), detok(b)


def pre_globalvoices(a: str, b: str):
    """對齊噪重（斷句切在逗號的子句碎片）→ 句尾逗號/冒號列丟。
    ponytail: 只做廉價過濾，錯位殘留靠低 cap 壓制；v3 評測 ja→zhtw 仍崩再上 LaBSE/COMET-QE。"""
    if FRAGMENT_END_RE.search(a) or FRAGMENT_END_RE.search(b):
        return None
    return a, b


def pre_opensub(a: str, b: str):
    """去掉 [音效]、(音效) 標記。"""
    return BRACKET_NOISE_RE.sub("", a).strip(), BRACKET_NOISE_RE.sub("", b).strip()


# TED2020 字幕：日文側 90% 沒有句末標點、中文側 39% 沒有——那是字幕排版慣例，
# 不是句子沒寫完。原封不動有兩種壞法：留著會教出「不加句號」，
# 交給 punct_asymmetric 則會丟掉 90% 的日文語料（＝唯一的文件級日文來源）。
# 折衷：看得出是完整句就補標點，看不出來就原樣退回讓過濾器丟。
JA_CONT_RE = re.compile(r"(?:から|けれど|けど|ので|のに|たり|つつ|ながら|ため|とき|こと|もの"
                        r"|[てでがはをにともやしのへ])$")
JA_END_RE = re.compile(r"(?:ます|ません|でした|ましょう|でしょう|ない|[すたるいうえおんかねよなだ])$")
# 中文碎片多半收在介詞／連詞／副詞，這些後面一定還有東西
ZH_CONT_RE = re.compile(r"(?:[比跟和與或及而並但把被從對給讓向往於以為因由當若雖則且也都還再又更最就才只]"
                        r"|以及|因為|所以|但是|不過|然後|而且|如果|雖然|由於)$")


TERM_OF = {"ja": "。", "zhtw": "。", "en": "."}


def restore_punct(s: str, lang: str) -> str:
    """句末缺標點但看得出是完整句 → 補句號；判斷不出來就原樣退回。"""
    if not s or s.endswith(END_PUNCT):
        return s
    # 收在逗號、破折號、括號等非終止標點的，本來就是沒寫完的碎片
    if not (s[-1].isalnum() or CJK_RE.match(s[-1]) or KANA_RE.match(s[-1])):
        return s
    if lang == "ja":
        return s + "。" if JA_END_RE.search(s) and not JA_CONT_RE.search(s) else s
    if lang == "zhtw":
        return s if ZH_CONT_RE.search(s) else s + "。"
    return s + TERM_OF[lang]


def restore_pair(a: str, b: str, la: str, lb: str):
    """拿另一側的句末標點當「這裡是句界」的證據再補；兩側都沒有＝沒有證據，整列丟。

    只憑單側猜會補錯（中文碎片「…開車到東邊」照補句號），
    而字幕語料兩側皆無標點的行（TED ja-zhtw 佔 37%）正是在教模型不加句號。"""
    pa, pb = a.endswith(END_PUNCT), b.endswith(END_PUNCT)
    if pa == pb:
        return (a, b) if pa else None
    return (a, restore_punct(b, lb)) if pa else (restore_punct(a, la), b)


def pre_ted_en_zhtw(a: str, b: str):
    return restore_pair(a, b, "en", "zhtw")


def pre_ted_en_ja(a: str, b: str):
    return restore_pair(a, b, "en", "ja")


def pre_ted_ja_zhtw(a: str, b: str):
    return restore_pair(a, b, "ja", "zhtw")


PREPROC = {
    "ted2020.en-zhtw.tsv": pre_ted_en_zhtw,
    "ted2020.en-ja.tsv": pre_ted_en_ja,
    "ted2020.ja-zhtw.tsv": pre_ted_ja_zhtw,
    "kde4.en-zhtw.tsv": pre_kde4, "kde4.ja-zhtw.tsv": pre_kde4,
    "globalvoices.en-zhtw.tsv": pre_globalvoices,
    "globalvoices.ja-zhtw.tsv": pre_globalvoices,
    "opensub.en-ja.tsv": pre_opensub, "opensub.en-zhtw.tsv": pre_opensub,
}


# --- v4 噪音過濾（事故根因，見 docs/RESEARCH-v4.md §2）------------------------
END_PUNCT = tuple("。．.!?！？」』）)…\"'~～")
# KDE4 等 tokenized 語料 detok 沒清乾淨的殘跡：`http: // www.`、`API_ KEY`、`NT $ 3,480`
STRAY_SPACE_RE = re.compile(
    r"[A-Za-z0-9一-鿿][ ]+[_/?=:%&][ ]?|[_/?=][ ]+[A-Za-z0-9一-鿿]")


def punct_asymmetric(a: str, b: str) -> bool:
    """一側有句末標點、另一側沒有 → 不對稱噪音（TED/字幕講稿無標點對上有標點的原文）。
    兩側都沒標點是風格一致，留著；模型學到的是「照抄來源的標點風格」而非「不加標點」。"""
    return a.endswith(END_PUNCT) != b.endswith(END_PUNCT)


def is_noise_pair(a: str, b: str) -> bool:
    """歌詞符號，或任一側是整句自我重複（字幕對齊錯誤的典型樣態）。"""
    if MUSIC_RE.search(a) or MUSIC_RE.search(b):
        return True
    for s in (a, b):
        h = len(s) // 2
        if len(s) >= 6 and len(s) % 2 == 0 and s[:h] == s[h:]:
            return True
    return False


def eff_len(s: str) -> int:
    """CJK 字元算 2，其餘算 1，讓中日英長度可比。"""
    return sum(2 if CJK_RE.match(c) or KANA_RE.match(c) else 1 for c in s)


def valid_lang(s: str, lang: str) -> bool:
    if lang == "en":
        return not CJK_RE.search(s) and not KANA_RE.search(s) and \
            sum(c.isascii() for c in s) / len(s) > 0.8
    if lang == "ja":
        return bool(KANA_RE.search(s))
    if lang == "zhtw":
        return bool(CJK_RE.search(s)) and not KANA_RE.search(s)
    raise ValueError(lang)


def has_simplified(s: str) -> bool:
    return cc_s2tw.convert(s) != s


def load_eval_lines() -> set[str]:
    """B1 污染閘：eval 基準的所有個別語言行（normalized，含 s2twp 變體）。缺檔即死。"""
    f = ROOT / "data" / "eval_lines.txt"
    if not f.exists():
        sys.exit("!! 缺 data/eval_lines.txt — 先跑 `uv run python scripts/dump_eval_lines.py`"
                 "（污染閘為強制，防訓練資料撞 eval 基準）")
    lines = set(f.read_text(encoding="utf-8").rstrip("\n").split("\n"))
    print(f"  eval contamination gate: {len(lines):,} lines loaded")
    return lines


def load_corpus(fname: str, l1: str, l2: str, stats: Counter, eval_lines: set[str]):
    """讀 TSV → 清洗 → 回傳 [(lineno, a, b)]，b 端若為 zh 已統一台灣正體。

    保留原始行號是為了文件級重組：只有原檔連號的句子才能併成段落，
    中間被過濾掉的行會造成不連貫的假文件（見 build_docs）。"""
    path = RAW / fname
    if not path.exists():
        print(f"  !! missing {fname}, skipped")
        return []
    seen, out = set(), []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            stats[f"{fname}:total"] += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                stats[f"{fname}:bad_format"] += 1
                continue
            a, b = norm(parts[0]), norm(parts[1])
            a, b = strip_subtitle_noise(a), strip_subtitle_noise(b)
            pre = PREPROC.get(fname)
            if pre:
                r = pre(a, b)
                if r is None:
                    stats[f"{fname}:preproc_dropped"] += 1
                    continue
                a, b = r
            if not a or not b:
                stats[f"{fname}:empty"] += 1
                continue
            if a == b:  # 未翻譯列（任何語料都是垃圾）
                stats[f"{fname}:identical"] += 1
                continue
            if is_noise_pair(a, b):
                stats[f"{fname}:subtitle_noise"] += 1
                continue
            if punct_asymmetric(a, b):
                stats[f"{fname}:punct_asymmetric"] += 1
                continue
            if STRAY_SPACE_RE.search(a) or STRAY_SPACE_RE.search(b):
                stats[f"{fname}:stray_space"] += 1
                continue
            la, lb = eff_len(a), eff_len(b)
            if min(la, lb) < 3 or max(la, lb) > 1000:
                stats[f"{fname}:length"] += 1
                continue
            if max(la, lb) / min(la, lb) > 3.0:
                stats[f"{fname}:ratio"] += 1
                continue
            if l2 == "zhtw" and has_simplified(b):
                b = cc_s2twp.convert(b)
                stats[f"{fname}:s2twp_converted"] += 1
                if has_simplified(b):
                    stats[f"{fname}:simplified_residue"] += 1
                    continue
            if not valid_lang(a, l1) or not valid_lang(b, l2):
                stats[f"{fname}:lang_mismatch"] += 1
                continue
            # B1 污染閘：在 s2twp 轉換「之後」比對，字形已與 eval 變體一致
            if a in eval_lines or b in eval_lines:
                stats[f"{fname}:eval_contaminated"] += 1
                continue
            key = hash((a, b))
            if key in seen:
                stats[f"{fname}:dup"] += 1
                continue
            seen.add(key)
            out.append((lineno, a, b))
    stats[f"{fname}:kept"] = len(out)
    print(f"  {fname}: kept {len(out):,} / {stats[f'{fname}:total']:,}")
    return out


def build_docs(rows, rng):
    """[(lineno, a, b)]（原檔順序）→ [(a_doc, b_doc)]，只併原檔連號的句子。

    行號不連續代表中間的句子被過濾掉了，硬接起來會製造語意跳躍的假文件，
    模型會學到「輸出可以跳過內容」——正是我們要修的毛病，所以遇缺就斷開。"""
    docs, run = [], []
    for r in rows:
        # 兩側皆無句末標點的行（TED ja-zhtw 有 37%）躲得過 punct_asymmetric，
        # 但併成段落後就是「整段沒有句號」的範本，直接當斷點排除。
        if not (r[1].endswith(END_PUNCT) and r[2].endswith(END_PUNCT)):
            docs.extend(_chunk(run, rng))
            run = []
            continue
        if run and r[0] != run[-1][0] + 1:
            docs.extend(_chunk(run, rng))
            run = []
        run.append(r)
    docs.extend(_chunk(run, rng))
    return docs


def _chunk(run, rng):
    out, i = [], 0
    while len(run) - i >= DOC_MIN:
        seg = run[i:i + rng.randint(DOC_MIN, DOC_MAX)]
        out.append(("\n".join(x[1] for x in seg), "\n".join(x[2] for x in seg)))
        i += len(seg)
    return out


def waterfill(ceilings, budget):
    """把 budget 分給各來源：由上限小的開始，每輪平均分剩餘預算，
    取不滿的餘額自動流向後面較大的池子。回傳與 ceilings 同序的配額。

    不用「依序貪婪 + 單一來源上限」的原因見 docs/RESEARCH-v5.md F1：
    預算一小，前幾個來源就把額度吃光，後面的領域整個消失。
    """
    order = sorted(range(len(ceilings)), key=lambda i: ceilings[i])
    takes, left = [0] * len(ceilings), budget
    for n, i in enumerate(order):
        takes[i] = max(0, min(left // (len(order) - n), ceilings[i]))
        left -= takes[i]
    return takes


def load_replay(rng, n):
    """通用指令 replay 樣本（scripts/build_replay.py 產生）。缺檔就跳過並警告。"""
    if not REPLAY_FILE.exists():
        print(f"  !! 缺 {REPLAY_FILE.name} — 本次不混通用資料，模型會退化成翻譯函數"
              f"（先跑 `uv run python scripts/build_replay.py`）")
        return []
    rows = [json.loads(x) for x in REPLAY_FILE.read_text(encoding="utf-8").rstrip("\n").split("\n") if x]
    rng.shuffle(rows)
    if len(rows) < n:
        print(f"  !! replay 只有 {len(rows):,} 筆，少於目標 {n:,}")
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="每方向樣本上限")
    ap.add_argument("--doc-share", type=float, default=DOC_SHARE,
                    help="每方向預算裡文件級（多段落）樣本的佔比")
    ap.add_argument("--replay-share", type=float, default=REPLAY_SHARE,
                    help="通用指令 replay 佔最終 train 的比例（0 = 不混，會災難性遺忘）")
    args = ap.parse_args()

    rng = random.Random(SEED)
    stats = Counter()

    print("== loading & cleaning ==")
    eval_lines = load_eval_lines()
    pools = {}      # fname -> {(dl1,dl2): [(src,tgt)...]}  句級，兩方向不重疊
    doc_pools = {}  # fname -> {(dl1,dl2): [(src,tgt)...]}  文件級（多段落）
    for fname, l1, l2 in CORPORA:
        rows = load_corpus(fname, l1, l2, stats, eval_lines)
        if fname in DOC_SOURCES:  # 重組要原檔順序，必須在 shuffle 之前
            docs = build_docs(rows, rng)
            rng.shuffle(docs)
            half_d = len(docs) // 2
            doc_pools[fname] = {(l1, l2): docs[:half_d],
                                (l2, l1): [(b, a) for a, b in docs[half_d:]]}
            stats[f"{fname}:docs_built"] = len(docs)
        rows = [(a, b) for _, a, b in rows]
        rng.shuffle(rows)
        if fname in ONE_WAY:  # 噪聲來源全量只進指定方向（源側）
            d = ONE_WAY[fname]
            pools[fname] = {d: rows if d == (l1, l2)
                            else [(b, a) for a, b in rows]}
            continue
        half = len(rows) // 2
        pools[fname] = {
            (l1, l2): rows[:half],
            (l2, l1): [(b, a) for a, b in rows[half:]],
        }

    print("== sampling per direction ==")
    SFT.mkdir(parents=True, exist_ok=True)
    train, dev = [], []
    dir_counts, domain_counts = {}, {}
    doc_counts = {}
    for direction, (budget, sources) in RECIPES.items():
        if args.limit:
            budget = min(budget, args.limit)
        dname = f"{direction[0]}->{direction[1]}"
        doc_budget = int(budget * args.doc_share)
        sent_budget = budget - doc_budget
        hard = int(sent_budget * MAX_SHARE)  # 單一來源佔比上限（B2 防壟斷）
        picked = []
        doms = Counter()
        # 注水式分配。依序貪婪會讓排前面的來源把 room 吃光：v4 用 --limit 20000 跑時
        # sent_budget=17,000、hard=8,500，前兩個來源就填滿，每方向只剩 2 個領域，
        # en→zhtw 因此連 wiki 語域都沒看過（COMET −2.98，而同樣被砍量的 en→ja
        # 有 wikimatrix 反而 +1.78）。見 docs/RESEARCH-v5.md F1。
        ceilings = [min(cap or sent_budget, hard, len(pools.get(f, {}).get(direction, [])))
                    for f, cap in sources]
        for (fname, _), take in zip(sources, waterfill(ceilings, sent_budget)):
            picked.extend(pools.get(fname, {}).get(direction, [])[:take])
            stats[f"dir:{dname}:{fname}"] = take
            doms[DOMAIN[fname]] += take
        # 文件級：從該方向可用的 DOC_SOURCES 取滿 doc_budget。
        # 均分後小池子取不滿的餘額要讓給大池子（newscomm 只有 59 篇，
        # 平均分會讓 en->ja 卡在 1,559 而非 3,000），故由小到大逐一水位填。
        docs = []
        avail = [f for f, _ in sources if f in doc_pools
                 and doc_pools[f].get(direction)]
        for fname, take in zip(avail, waterfill(
                [len(doc_pools[f][direction]) for f in avail], doc_budget)):
            docs.extend(doc_pools[fname][direction][:take])
            stats[f"dir:{dname}:{fname}:doc"] = take
        doc_counts[dname] = len(docs)
        picked.extend(docs)
        rng.shuffle(picked)
        dev.extend((direction, s, t) for s, t in picked[:DEV_PER_DIR])
        train.extend((direction, s, t) for s, t in picked[DEV_PER_DIR:])
        dir_counts[dname] = len(picked)
        domain_counts[dname] = dict(doms)
        print(f"  {dname}: {len(picked):,} (文件級 {len(docs):,})  {dict(doms)}")

    rng.shuffle(train)

    def to_rec(row):
        (src_l, tgt_l), src, tgt = row
        return {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{rng.choice(TEMPLATES[tgt_l])}\n{src}"},
            {"role": "assistant", "content": tgt},
        ]}

    # 通用 replay：翻譯樣本佔 (1-share)，據此反推 replay 筆數
    print("== replay (通用指令) ==")
    n_replay = int(len(train) * args.replay_share / max(1 - args.replay_share, 1e-6))
    replay = load_replay(rng, n_replay)
    records = [to_rec(r) for r in train] + replay
    rng.shuffle(records)
    got = len(replay) / max(len(records), 1) * 100
    print(f"  replay {len(replay):,} / total {len(records):,} = {got:.1f}%")

    def dump(path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {path} ({len(rows):,})")

    print("== writing jsonl ==")
    dump(SFT / "train.jsonl", records)
    dump(SFT / "dev.jsonl", [to_rec(r) for r in dev])

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump({"directions": dir_counts, "doc_level": doc_counts,
                   "replay": {"n": len(replay), "realized_pct": round(got, 2)},
                   "domains": domain_counts, "details": dict(stats)},
                  f, ensure_ascii=False, indent=2)
    print("stats -> results/data_stats.json")


if __name__ == "__main__":
    main()
