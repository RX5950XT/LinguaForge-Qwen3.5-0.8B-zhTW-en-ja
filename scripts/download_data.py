"""Phase 1: 下載平行語料 → data/raw/*.tsv（兩欄，tab 分隔，檔名標明語對與方向）。

來源：
- HF zetavg/coct-en-zh-tw-translations-twp-300k  (en↔zh-TW 原生台灣正體)
- HF Helsinki-NLP/opus-100  en-zh / en-ja        (zh 為簡體，後續 s2twp)
- HF Verah/JParaCrawl-Filtered  en-ja            (LLM 過濾 JParaCrawl v3，取兩 LLM 皆過)
- OPUS TED2020 en-zh_tw / ja-zh_tw / en-ja       (原生繁體字幕 + 乾淨 en-ja)
- OPUS WikiMatrix ja-zh / en-ja                  (維基書面體，貼 FLORES domain)
- OPUS Tatoeba / News-Commentary en-ja           (母語者短句 + 新聞書面體)
- OPUS News-Commentary ja-zh                     (書面體，補 zhtw↔ja 品質)
- OPUS OpenSubtitles ja-zh_tw                    (繁體字幕，口語、較噪)

v3 新增（Phase B3，依多領域診斷對症）：
- OPUS GlobalVoices en-zht / jp-zht              (新聞，原生繁體；對齊噪，prepare 過濾)
- OPUS KDE4 en-zh_TW / ja-zh_TW                  (IT 在地化，原生台灣繁體；含佔位符，prepare 過濾)
- OPUS KFTT en-ja                                (京都維基百科體，CC-BY-SA-3.0)
- OPUS OpenSubtitles en-ja v2024 / en-zh_tw v2018 (口語源側，救 WMT into-English；僅入 →en 方向)
- GitHub MTNT ja-en                              (Reddit 口語噪聲，僅入 ja→en 源側)

日文方向（en↔ja、zhtw↔ja）原本全靠雜訊多的 OPUS-100，改用上列乾淨書面體語料重訓。
"""

import io
import sys
import zipfile
from pathlib import Path

import requests

RAW = Path(__file__).parent.parent / "data" / "raw"
OPUS_JOBS = [
    # (name, url, lang1_ext, lang2_ext, out_name)
    ("TED2020 en-zhtw",
     "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-zh_tw.txt.zip",
     "en", "zh_tw", "ted2020.en-zhtw.tsv"),
    ("TED2020 ja-zhtw",
     "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/ja-zh_tw.txt.zip",
     "ja", "zh_tw", "ted2020.ja-zhtw.tsv"),
    ("OpenSubtitles ja-zhtw",
     "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/ja-zh_tw.txt.zip",
     "ja", "zh_tw", "opensub.ja-zhtw.tsv"),
    ("WikiMatrix ja-zh",
     "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/ja-zh.txt.zip",
     "ja", "zh", "wikimatrix.ja-zh.tsv"),
    # --- 乾淨書面體 en-ja（取代 OPUS-100 的雜訊）---
    ("WikiMatrix en-ja",
     "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-ja.txt.zip",
     "en", "ja", "wikimatrix.en-ja.tsv"),
    ("TED2020 en-ja",
     "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-ja.txt.zip",
     "en", "ja", "ted2020.en-ja.tsv"),
    ("Tatoeba en-ja",
     "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-ja.txt.zip",
     "en", "ja", "tatoeba.en-ja.tsv"),
    ("News-Commentary en-ja",
     "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/en-ja.txt.zip",
     "en", "ja", "newscomm.en-ja.tsv"),
    ("News-Commentary ja-zh",
     "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/ja-zh.txt.zip",
     "ja", "zh", "newscomm.ja-zh.tsv"),
    # --- v3 Phase B3（注意 GlobalVoices 語碼是 jp/zht，不是 ja/zh）---
    ("GlobalVoices en-zhtw",
     "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-zht.txt.zip",
     "en", "zht", "globalvoices.en-zhtw.tsv"),
    ("GlobalVoices ja-zhtw",
     "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/jp-zht.txt.zip",
     "jp", "zht", "globalvoices.ja-zhtw.tsv"),
    ("KDE4 en-zhtw",
     "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/en-zh_TW.txt.zip",
     "en", "zh_TW", "kde4.en-zhtw.tsv"),
    ("KDE4 ja-zhtw",
     "https://object.pouta.csc.fi/OPUS-KDE4/v2/moses/ja-zh_TW.txt.zip",
     "ja", "zh_TW", "kde4.ja-zhtw.tsv"),
    ("KFTT en-ja",
     "https://object.pouta.csc.fi/OPUS-KFTT/v1.0/moses/en-ja.txt.zip",
     "en", "ja", "kftt.en-ja.tsv"),
    ("OpenSubtitles en-ja",
     "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-ja.txt.zip",
     "en", "ja", "opensub.en-ja.tsv"),
    ("OpenSubtitles en-zhtw",
     "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-zh_tw.txt.zip",
     "en", "zh_tw", "opensub.en-zhtw.tsv"),
]

MTNT_URL = "https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz"


def download_mtnt():
    """MTNT ja-en train：TSV 格式 comment_id\\tsource(ja)\\ttarget(en)。
    輸出統一為 (en, ja) 欄序對齊其他 en-ja 檔；src==tgt 未翻譯列在此就丟。"""
    out = RAW / "mtnt.ja-en.tsv"
    if out.exists():
        print("[skip] mtnt.ja-en.tsv already exists")
        return
    import tarfile
    print(f"[mtnt] <- {MTNT_URL}")
    r = requests.get(MTNT_URL, timeout=600)
    r.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
        member = next(m for m in tf.getmembers()
                      if m.name.endswith("train/train.ja-en.tsv"))
        lines = tf.extractfile(member).read().decode("utf-8").splitlines()
    pairs = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) == 3 and parts[1] != parts[2]:  # 丟未翻譯（src==tgt）
            pairs.append((parts[2], parts[1]))  # (en, ja)
    write_tsv(out, pairs)


def clean(s: str) -> str:
    return s.replace("\t", " ").replace("\r", "").strip()


def write_tsv(path: Path, pairs) -> int:
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for a, b in pairs:
            a, b = clean(a), clean(b)
            if a and b:
                f.write(f"{a}\t{b}\n")
                n += 1
    print(f"  {path.name}: {n:,} pairs")
    return n


def download_opus(name, url, l1, l2, out_name):
    out = RAW / out_name
    if out.exists():
        print(f"[skip] {out_name} already exists")
        return
    print(f"[opus] {name} <- {url}")
    r = requests.get(url, timeout=600)
    if r.status_code != 200:
        print(f"  !! HTTP {r.status_code}, skipped")
        return
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    f1 = next(n for n in zf.namelist() if n.endswith(f".{l1}"))
    f2 = next(n for n in zf.namelist() if n.endswith(f".{l2}"))
    lines1 = zf.read(f1).decode("utf-8").splitlines()
    lines2 = zf.read(f2).decode("utf-8").splitlines()
    assert len(lines1) == len(lines2), f"line count mismatch {len(lines1)} vs {len(lines2)}"
    write_tsv(out, zip(lines1, lines2))


def download_hf():
    from datasets import load_dataset

    out = RAW / "coct.en-zhtw.tsv"
    if not out.exists():
        print("[hf] zetavg/coct-en-zh-tw-translations-twp-300k")
        ds = load_dataset("zetavg/coct-en-zh-tw-translations-twp-300k", split="train")
        cols = ds.column_names
        print(f"  columns: {cols}")
        en_col = next(c for c in cols if "en" in c.lower())
        zh_col = next(c for c in cols if c != en_col)
        write_tsv(out, ((row[en_col], row[zh_col]) for row in ds))
    else:
        print("[skip] coct.en-zhtw.tsv already exists")

    for pair, l1, l2 in [("en-ja", "en", "ja"), ("en-zh", "en", "zh")]:
        out = RAW / f"opus100.{pair}.tsv"
        if out.exists():
            print(f"[skip] opus100.{pair}.tsv already exists")
            continue
        print(f"[hf] opus-100 {pair}")
        ds = load_dataset("Helsinki-NLP/opus-100", pair, split="train")
        write_tsv(out, ((row["translation"][l1], row["translation"][l2]) for row in ds))

    # JParaCrawl v3（LLM 過濾）：只取兩個 judge 都 accept 的列 → 最乾淨子集
    out = RAW / "jparacrawl.en-ja.tsv"
    if not out.exists():
        print("[hf] Verah/JParaCrawl-Filtered (both models accepted)")
        ds = load_dataset("Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus",
                          split="train")
        rows = ((r["english"], r["japanese"]) for r in ds
                if r["model1_accepted"] and r["model2_accepted"])
        write_tsv(out, rows)
    else:
        print("[skip] jparacrawl.en-ja.tsv already exists")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    download_hf()
    try:
        download_mtnt()
    except Exception as e:
        print(f"  !! MTNT failed: {e}")
    for job in OPUS_JOBS:
        try:
            download_opus(*job)
        except Exception as e:  # 單一語料失敗不擋整批，最後總結
            print(f"  !! {job[0]} failed: {e}")
    print("\nDONE. files in data/raw:")
    for p in sorted(RAW.glob("*.tsv")):
        print(f"  {p.name}  {p.stat().st_size / 1024**2:.1f} MB")
