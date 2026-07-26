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

CTRL_RE = re.compile("[\u0000-\u0008\u000b-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
KANA_RE = re.compile("[぀-ヿ]")
CJK_RE = re.compile("[一-鿿㐀-䶿]")

cc_s2twp = OpenCC("s2twp")
cc_s2t = OpenCC("s2t")


SPEAKER_PAREN_RE = re.compile(r"^\s*[（(][^（）()]{1,25}[）)]\s*")
SPEAKER_TAG_RE = re.compile(r"^\s*[A-Z][A-Za-z]{0,10}\s*[:：]{1,2}\s*")
DASH_RE = re.compile(r"^\s*[-－—]+\s*")
MUSIC_RE = re.compile(r"[♪♫♬]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = CTRL_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


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


PREPROC = {
    "kde4.en-zhtw.tsv": pre_kde4, "kde4.ja-zhtw.tsv": pre_kde4,
    "globalvoices.en-zhtw.tsv": pre_globalvoices,
    "globalvoices.ja-zhtw.tsv": pre_globalvoices,
    "opensub.en-ja.tsv": pre_opensub, "opensub.en-zhtw.tsv": pre_opensub,
}


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
    return cc_s2t.convert(s) != s


def load_eval_lines() -> set[str]:
    """B1 污染閘：eval 基準的所有個別語言行（normalized，含 s2twp 變體）。缺檔即死。"""
    f = ROOT / "data" / "eval_lines.txt"
    if not f.exists():
        sys.exit("!! 缺 data/eval_lines.txt — 先跑 `uv run python scripts/dump_eval_lines.py`"
                 "（污染閘為強制，防訓練資料撞 eval 基準）")
    lines = set(f.read_text(encoding="utf-8").splitlines())
    print(f"  eval contamination gate: {len(lines):,} lines loaded")
    return lines


def load_corpus(fname: str, l1: str, l2: str, stats: Counter, eval_lines: set[str]):
    """讀 TSV → 清洗 → 回傳 [(a, b)]，b 端若為 zh 已統一台灣正體。"""
    path = RAW / fname
    if not path.exists():
        print(f"  !! missing {fname}, skipped")
        return []
    seen, out = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
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
            out.append((a, b))
    stats[f"{fname}:kept"] = len(out)
    print(f"  {fname}: kept {len(out):,} / {stats[f'{fname}:total']:,}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="每方向樣本上限")
    args = ap.parse_args()

    rng = random.Random(SEED)
    stats = Counter()

    print("== loading & cleaning ==")
    eval_lines = load_eval_lines()
    pools = {}  # fname -> {(dl1,dl2): [(src,tgt)...]} 兩方向不重疊
    for fname, l1, l2 in CORPORA:
        rows = load_corpus(fname, l1, l2, stats, eval_lines)
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
    for direction, (budget, sources) in RECIPES.items():
        if args.limit:
            budget = min(budget, args.limit)
        dname = f"{direction[0]}->{direction[1]}"
        hard = int(budget * MAX_SHARE)  # 單一來源佔比上限（B2 防壟斷）
        picked = []
        doms = Counter()
        for fname, cap in sources:
            room = budget - len(picked)
            if room <= 0:
                break
            pool = pools.get(fname, {}).get(direction, [])
            take = min(room, cap if cap else room, hard, len(pool))
            picked.extend(pool[:take])
            stats[f"dir:{dname}:{fname}"] = take
            doms[DOMAIN[fname]] += take
        rng.shuffle(picked)
        dev.extend((direction, s, t) for s, t in picked[:DEV_PER_DIR])
        train.extend((direction, s, t) for s, t in picked[DEV_PER_DIR:])
        dir_counts[dname] = len(picked)
        domain_counts[dname] = dict(doms)
        print(f"  {dname}: {len(picked):,}  {dict(doms)}")

    rng.shuffle(train)

    def dump(path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for (src_l, tgt_l), src, tgt in rows:
                instr = rng.choice(TEMPLATES[tgt_l])
                rec = {"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"{instr}\n{src}"},
                    {"role": "assistant", "content": tgt},
                ]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {path} ({len(rows):,})")

    print("== writing jsonl ==")
    dump(SFT / "train.jsonl", train)
    dump(SFT / "dev.jsonl", dev)

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump({"directions": dir_counts, "domains": domain_counts,
                   "details": dict(stats)}, f, ensure_ascii=False, indent=2)
    print("stats -> results/data_stats.json")


if __name__ == "__main__":
    main()
