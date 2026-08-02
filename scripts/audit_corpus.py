"""SFT 訓練集 target 側污染稽核（A/B/C/D 四類），輸出量化報告 + 抽樣。

只讀不寫資料，產出 results/audit/corpus.json 與主控台摘要。

  A 標籤前綴：target 行首出現原文沒有的標籤／編號／圖說／敘事框架詞
  B 專名遺失：source 有拉丁專名（連續大寫詞、含數字型號）但 target 完全沒有
  C 憑空年份：target 有四位數年份而 source 沒有
  D 多行壓扁：source 多行但 target 單行

用法：
  uv run python scripts/audit_corpus.py                    # 稽核 data/sft/train.jsonl
  uv run python scripts/audit_corpus.py --raw              # 另掃 data/raw/*.tsv（分來源）
  uv run python scripts/audit_corpus.py --samples 10
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN = ROOT / "data" / "sft" / "train.jsonl"
RAW = ROOT / "data" / "raw"

# 指令模板 → 目標語言（與 prepare_data.TEMPLATES 對齊）
INSTR2TGT = {
    "翻譯成繁體中文：": "zhtw", "翻譯成臺灣正體中文：": "zhtw",
    "Translate to Traditional Chinese (Taiwan):": "zhtw",
    "台湾の繁体字中国語に翻訳してください：": "zhtw",
    "翻譯成英文：": "en", "Translate to English:": "en", "英語に翻訳してください：": "en",
    "翻譯成日文：": "ja", "Translate to Japanese:": "ja", "日本語に翻訳してください：": "ja",
}

# A 類樣式。命名與 bench_defects.py 一致，方便前後對照。
A_PATTERNS = [
    ("label:說明", re.compile(r"^[ \t]*說明[：:]")),
    ("label:註", re.compile(r"^[ \t]*(?:註解|註釋|註|備註|注意|提示)[：:]")),
    ("label:問答", re.compile(r"^[ \t]*(?:問|答|Q|A)[：:]")),
    ("label:譯者", re.compile(r"^[ \t]*(?:譯者|譯文|翻譯者|翻譯員|譯員)[：:]")),
    ("label:結構", re.compile(r"^[ \t]*(?:標題|內容|摘要|總結|結論|原文|正文|前言)[：:]")),
    ("label:Note", re.compile(r"^[ \t]*(?:Note|NOTE|Caption)[：:]")),
    ("enum:數字", re.compile(r"^[ \t]*\d{1,2}[.、)]\s")),
    ("figure:圖為", re.compile(r"^[ \t]*(?:圖為|照片為|圖片為|上圖為|圖說)")),
    ("figure:圖N", re.compile(r"^[ \t]*圖\s*\d*\s*[.號：:]")),
    ("select:選擇", re.compile(r"^[ \t]*選擇[：:]")),
    ("narrate:故事說", re.compile(r"^[ \t]*(?:故事說|傳說|相傳)[，,]")),
    ("narrate:據報導", re.compile(r"^[ \t]*(?:據報導|據報道|根據報導|根據報道|報導說|報道稱|據悉)[，,]?")),
]
# 對應的來源側樣式：判「target 憑空多出來」還是「source 本來就有」
A_SRC_PATTERNS = re.compile(
    r"^[ \t]*(?:"
    r"\d{1,2}[.、)]\s"                                     # 1. / 2)
    r"|(?:Note|NOTE|Caption|Q|A|Fig(?:ure)?\.?\s*\d*)\s*[：:]"
    r"|(?:注|注意|説明|説明|図|写真|問|答|キャプション)\s*[：:]"
    r"|說明[：:]|註[：:]|問[：:]|答[：:]|圖\s*\d*\s*[.號：:]|選擇[：:]"
    r"|(?:It is said that|Legend has it|According to reports|Reportedly|Pictured)"
    r"|(?:と言われ|によると|伝説)"
    r")", re.IGNORECASE)

# B 類：拉丁專名 —— 連續大寫開頭詞、全大寫縮寫、含數字型號（H200 / GLM 5.5 / 2nm）
PROPER = re.compile(r"\b(?:[A-Z][a-zA-Z]*[0-9]+[a-zA-Z0-9]*|[A-Z]{2,}[0-9]*|[A-Z][a-z]{2,})\b")
# 寬版 PROPER 對 →zhtw/→ja 會大量誤報：`Japan`→日本、`Mr. Ovia`→奧維亞先生
# 都是正確意譯，不是「吃掉專名」。嚴格版只收「本來就該原樣保留」的字串：
# 全大寫縮寫（NVIDIA/TSMC/GLM）與含數字型號（H200/HBM3e/2nm）。
PROPER_STRICT = re.compile(r"\b(?:[A-Z]{2,}[0-9]*|[A-Za-z]{1,10}[0-9]+[a-zA-Z]*)\b")
STRICT_STOP = {"I", "A", "OK", "TV", "US", "UK", "UN", "AM", "PM", "MR", "DR", "AND",
               "THE", "OF", "IN", "TO", "IS", "IT", "NO", "ON", "OR", "SO", "BE"}
# 句首大寫的普通字不算專名（The / This / We…），扣掉最常見的英文句首詞
STOP = {"The", "This", "That", "These", "Those", "There", "Then", "They", "Their",
        "And", "But", "For", "You", "Your", "Our", "His", "Her", "Its", "She", "One",
        "When", "What", "Where", "Which", "While", "With", "Who", "Why", "How",
        "After", "Before", "All", "Any", "Are", "But", "Can", "Did", "Does", "Each",
        "Even", "Every", "From", "Had", "Has", "Have", "Here", "However", "Into",
        "Just", "Like", "Many", "May", "More", "Most", "Not", "Now", "Only", "Other",
        "Some", "Such", "Than", "Was", "Were", "Will", "Would", "Should", "Could",
        "Also", "Because", "Been", "Being", "Between", "Both", "During", "Under",
        "Over", "Same", "Since", "Still", "Thus", "Too", "Very", "Well", "Yes", "Yet"}
YEAR = re.compile(r"(?:1[89]|20)\d{2}")


def direction_of(user: str, assistant: str):
    """(src_lang, tgt_lang, 原文)；抓不到模板回 None。"""
    head, _, body = user.partition("\n")
    tgt = INSTR2TGT.get(head.strip())
    if tgt is None:
        return None
    return (lang_of(body), tgt, body)


_JA = re.compile(r"[぀-ゟ゠-ヿ]")
_HAN = re.compile(r"[一-鿿]")


def lang_of(text: str) -> str:
    if _JA.search(text):
        return "ja"
    if _HAN.search(text):
        return "zhtw"
    return "en"


def proper_nouns(text: str) -> set[str]:
    return {m for m in PROPER.findall(text) if m not in STOP and len(m) >= 2}


def proper_strict(text: str) -> set[str]:
    return {m for m in PROPER_STRICT.findall(text)
            if m.upper() not in STRICT_STOP and len(m) >= 2}


def audit_pair(src: str, tgt: str) -> list[str]:
    hits = []
    src_head = src.split("\n", 1)[0]
    for name, pat in A_PATTERNS:
        if pat.match(tgt) and not A_SRC_PATTERNS.match(src_head):
            hits.append("A/" + name)
    props = proper_nouns(src)
    if props and not any(p in tgt for p in props):
        hits.append("B/專名全失")
    strict = proper_strict(src)
    if strict and any(p not in tgt for p in strict):
        hits.append("B/型名遺失")
    ghost = set(YEAR.findall(tgt)) - set(YEAR.findall(src))
    if ghost:
        hits.append("C/憑空年份")
    if "\n" in src.strip() and "\n" not in tgt.strip():
        hits.append("D/多行壓扁")
    return hits


def report(counts, totals, samples, title, key_name):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    for k in sorted(totals, key=lambda x: -totals[x]):
        n = totals[k]
        rows = [(p, c) for p, c in counts[k].items()]
        if not rows:
            print(f"\n-- {key_name}={k}  n={n:,}  （零命中）")
            continue
        print(f"\n-- {key_name}={k}  n={n:,}")
        for p, c in sorted(rows, key=lambda x: -x[1]):
            print(f"     {p:<20} {c:>8,}  {c / n * 100:6.3f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--raw", action="store_true", help="另掃 data/raw/*.tsv 分來源統計")
    args = ap.parse_args()

    counts = defaultdict(Counter)      # direction -> pattern -> n
    totals = Counter()                 # direction -> n
    samples = defaultdict(list)        # pattern -> [(src, tgt)]
    prop_stat = Counter()              # 專名分母/分子
    multiline = Counter()
    skipped = 0

    with open(args.train, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            msgs = {m["role"]: m["content"] for m in rec["messages"]}
            if "user" not in msgs or "assistant" not in msgs:
                continue
            d = direction_of(msgs["user"], msgs["assistant"])
            if d is None:
                skipped += 1        # replay（非翻譯任務）
                continue
            src_l, tgt_l, src = d
            tgt = msgs["assistant"]
            key = f"{src_l}->{tgt_l}"
            totals[key] += 1
            if proper_nouns(src):
                prop_stat[f"{key}:有專名"] += 1
            if proper_strict(src):
                prop_stat[f"{key}:有型名"] += 1
            if "\n" in src.strip():
                multiline[f"{key}:多行來源"] += 1
            for h in audit_pair(src, tgt):
                counts[key][h] += 1
                if h == "B/專名全失":
                    prop_stat[f"{key}:全失"] += 1
                if h == "B/型名遺失":
                    prop_stat[f"{key}:型名失"] += 1
                if h == "D/多行壓扁":
                    multiline[f"{key}:壓扁"] += 1
                if len(samples[h]) < args.samples:
                    samples[h].append({"direction": key, "src": src[:300], "tgt": tgt[:300]})

    report(counts, totals, samples, f"train.jsonl target 側污染率（replay 略過 {skipped:,} 筆）", "方向")

    print(f"\n{'=' * 70}\nB 類分母（source 含拉丁專名的樣本裡，target 完全沒有的比例）\n{'=' * 70}")
    for key in sorted(totals):
        has, lost = prop_stat[f"{key}:有專名"], prop_stat[f"{key}:全失"]
        hs, ls = prop_stat[f"{key}:有型名"], prop_stat[f"{key}:型名失"]
        if has:
            print(f"  {key:<12} 寬版 有 {has:>8,} 全失 {lost:>7,} {lost / has * 100:6.2f}%"
                  f"   |  嚴格(縮寫/型號) 有 {hs:>7,} 遺失 {ls:>7,} "
                  f"{(ls / hs * 100) if hs else 0:6.2f}%")

    print(f"\n{'=' * 70}\nD 類分母（source 多行的樣本裡，target 被壓成單行的比例）\n{'=' * 70}")
    for key in sorted(totals):
        has, flat = multiline[f"{key}:多行來源"], multiline[f"{key}:壓扁"]
        if has:
            print(f"  {key:<12} 多行來源 {has:>8,}  壓扁 {flat:>7,}  {flat / has * 100:6.2f}%")

    print(f"\n{'=' * 70}\n抽樣（每樣式最多 {args.samples} 筆）\n{'=' * 70}")
    for p in sorted(samples):
        print(f"\n### {p}")
        for s in samples[p]:
            print(f"  [{s['direction']}]")
            print(f"    SRC {s['src']!r}")
            print(f"    TGT {s['tgt']!r}")

    out = {"totals": dict(totals), "counts": {k: dict(v) for k, v in counts.items()},
           "proper": dict(prop_stat), "multiline": dict(multiline),
           "samples": {k: v for k, v in samples.items()}, "replay_skipped": skipped}

    if args.raw:
        raw_counts, raw_totals = defaultdict(Counter), Counter()
        for tsv in sorted(RAW.glob("*.tsv")):
            with open(tsv, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 2:
                        continue
                    a, b = parts
                    for src, tgt in ((a, b), (b, a)):
                        key = f"{tsv.name}:->{lang_of(tgt)}"
                        raw_totals[key] += 1
                        for h in audit_pair(src, tgt):
                            raw_counts[key][h] += 1
        report(raw_counts, raw_totals, {}, "data/raw/*.tsv 分來源污染率（雙向各算一次）", "來源")
        out["raw"] = {"totals": dict(raw_totals),
                      "counts": {k: dict(v) for k, v in raw_counts.items()}}

    out_dir = ROOT / "results" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "corpus.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
