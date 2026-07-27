"""v4：建通用指令 replay 資料 → data/sft/replay.jsonl。

為什麼需要：v2/v3 訓練集 100% 是翻譯任務，模型學到的不是「翻譯」這個任務，
而是「chat 模板下的 user 內容一律翻譯」——問它「台灣最高山是哪座」它會把問題翻譯掉。
詳見 docs/RESEARCH-v4.md。Tower+ 的 SFT 配比是 22% 翻譯 / 78% 通用指令。

只收授權乾淨（Apache-2.0）且非機器翻譯的來源：
  OpenAssistant/oasst2   en/zh/ja 人工撰寫對話樹，取最高分的助理回覆
  CohereLabs/aya_dataset eng/jpn/zho 人工標註 prompt-completion
  llm-jp/oasst2-33k-ja   日文量體補充（oasst2 英文子集的 DeepL 日譯）

排除：aya_collection 的 traditional_chinese（機翻 Flan，序號被當文字翻譯成
「一,他們」「2. 我們的國家」，會毒化訓練）、TaiwanChat（CC-BY-NC，與本專案
Apache-2.0 衝突）。

zh 端一律 s2twp 轉台灣正體並複檢，避免 replay 反過來把簡體洩漏教回去。

用法：
  uv run python scripts/build_replay.py                 # 只用現成資料集（不需 GPU）
  uv run python scripts/build_replay.py --max-per-lang 40000
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "sft" / "replay.jsonl"

cc_s2twp = OpenCC("s2twp")
cc_s2tw = OpenCC("s2tw")

KANA_RE = re.compile("[぀-ヿ]")
CJK_RE = re.compile("[一-鿿]")
# str.splitlines() 會在這些字元上斷行，json.dumps 卻不跳脫它們 →
# 一筆記錄會被讀成好幾行、JSON 解析炸掉。寫入前一律換成真正的 \n。
LINESEP_RE = re.compile("[\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")
# 明顯是翻譯任務的 prompt 要剔除：replay 的目的是「非翻譯能力」，
# 混進翻譯指令等於沒有 replay 到
TRANSLATE_RE = re.compile(
    r"translat|翻譯|翻译|翻訳|译成|譯成|に翻訳", re.IGNORECASE)

LANG_OF = {"en": "en", "zh": "zhtw", "ja": "ja"}
AYA_LANG = {"English": "en", "Japanese": "ja", "Chinese": "zhtw"}


def clean(text: str, lang: str):
    """回傳清洗後文字，或 None 表示丟棄。"""
    t = LINESEP_RE.sub("\n", text or "").strip()
    if not (10 <= len(t) <= 4000):
        return None
    if lang == "zhtw":
        t = cc_s2twp.convert(t)
        if cc_s2tw.convert(t) != t:      # s2twp 後仍有簡體殘留 → 丟
            return None
        if not CJK_RE.search(t):
            return None
    elif lang == "ja":
        if not KANA_RE.search(t):
            return None
    return t


def emit(pairs, lang, stats, out):
    """[(user, assistant)] → messages 記錄。翻譯類 prompt 與清洗失敗者丟棄。"""
    for u, a in pairs:
        stats[f"{lang}:seen"] += 1
        if TRANSLATE_RE.search(u):
            stats[f"{lang}:drop_translation_prompt"] += 1
            continue
        cu, ca = clean(u, lang), clean(a, lang)
        if not cu or not ca:
            stats[f"{lang}:drop_clean"] += 1
            continue
        out.setdefault(lang, []).append(
            {"messages": [{"role": "user", "content": cu},
                          {"role": "assistant", "content": ca}]})


def from_oasst2(stats, out):
    """對話樹 → (prompter, 最高分 assistant 回覆) 配對，只取 en/zh/ja。"""
    from datasets import load_dataset

    print("== OpenAssistant/oasst2 ==")
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    by_id, children = {}, {}
    for r in ds:
        by_id[r["message_id"]] = r
        if r["parent_id"]:
            children.setdefault(r["parent_id"], []).append(r)

    def rank(r):  # rank=0 是同層最佳；None 排最後
        return r["rank"] if r["rank"] is not None else 99

    per_lang = {}
    for mid, r in by_id.items():
        if r["role"] != "prompter" or r["deleted"]:
            continue
        lang = LANG_OF.get((r["lang"] or "").split("-")[0])
        if not lang:
            continue
        replies = [c for c in children.get(mid, [])
                   if c["role"] == "assistant" and not c["deleted"]]
        if not replies:
            continue
        best = min(replies, key=rank)
        per_lang.setdefault(lang, []).append((r["text"], best["text"]))
    for lang, pairs in per_lang.items():
        print(f"  {lang}: {len(pairs):,} pairs")
        emit(pairs, lang, stats, out)


def from_aya(stats, out):
    from datasets import load_dataset

    print("== CohereLabs/aya_dataset ==")
    ds = load_dataset("CohereLabs/aya_dataset", split="train")
    per_lang = {}
    for r in ds:
        lang = AYA_LANG.get(r["language"])
        if lang:
            per_lang.setdefault(lang, []).append((r["inputs"], r["targets"]))
    for lang, pairs in per_lang.items():
        print(f"  {lang}: {len(pairs):,} pairs")
        emit(pairs, lang, stats, out)


def from_oasst2_ja(stats, out):
    from datasets import load_dataset

    print("== llm-jp/oasst2-33k-ja ==")
    ds = load_dataset("llm-jp/oasst2-33k-ja", split="train")
    pairs = []
    for r in ds:
        conv = r["conversations"]
        for i in range(len(conv) - 1):
            if conv[i]["role"] in ("user", "prompter") and \
                    conv[i + 1]["role"] == "assistant":
                pairs.append((conv[i]["content"], conv[i + 1]["content"]))
    print(f"  ja: {len(pairs):,} pairs")
    emit(pairs, "ja", stats, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-lang", type=int, default=None,
                    help="每語言上限（不設＝全收）")
    args = ap.parse_args()

    stats, out = Counter(), {}
    for fn in (from_oasst2, from_aya, from_oasst2_ja):
        try:
            fn(stats, out)
        except Exception as e:   # 單一來源不可用不該讓整批失敗
            print(f"  !! {fn.__name__} failed: {type(e).__name__}: {e}")

    if not out:
        raise SystemExit("!! 沒有取得任何 replay 樣本，請檢查網路 / HF 快取")

    import random
    rng = random.Random(42)
    rows = []
    print("== 結果 ==")
    for lang in ("en", "ja", "zhtw"):
        got = out.get(lang, [])
        # 同一 prompt 可能來自多個來源 → 去重
        seen, uniq = set(), []
        for r in got:
            k = r["messages"][0]["content"]
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        rng.shuffle(uniq)
        if args.max_per_lang:
            uniq = uniq[:args.max_per_lang]
        rows.extend(uniq)
        print(f"  {lang}: {len(uniq):,}  (原始 {len(got):,}, 去重 -{len(got)-len(uniq):,})")

    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT} ({len(rows):,})")
    print(f"drops: { {k: v for k, v in stats.items() if 'drop' in k} }")


if __name__ == "__main__":
    main()
