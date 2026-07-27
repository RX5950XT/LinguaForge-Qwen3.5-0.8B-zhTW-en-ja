"""能力面板評測 — 補上舊 evaluate.py 測不到的三個維度。

舊 evaluate.py 的五個基準全是「單句翻譯」，v2/v3 在上面 COMET 84+ 但實際上：
長篇只翻前兩段就停、任何非翻譯指令都被當成待翻譯文本。本腳本就是為了讓這種
退化在數字上現形。

三軸（--axis）：
  doc      文件級翻譯（WMT24++ 按 document_id 聚合，六方向 en-pivot）
           → chrF++ / 完整度 / 漏譯 / 重複 / 簡體洩漏
  ifeval   可驗證指令跟隨（自建 zh-TW / ja / en，規則判分，IFEval 作法）
           → pass rate
  general  通用能力保留（唯一答案問答 + 「退化成翻譯機」自動偵測）
           → 正確率 / 翻譯機率

用法：
  uv run python scripts/eval_capability.py --tag base
  uv run python scripts/eval_capability.py --tag v3 --adapter outputs/sft-v3
  uv run python scripts/eval_capability.py --tag v3 --adapter outputs/sft-v3 --axis doc --docs 30
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from opencc import OpenCC
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).parent.parent
MODEL_ID = "Qwen/Qwen3.5-0.8B"
SYSTEM = "You are a professional translator."
INSTR = {"zhtw": "翻譯成繁體中文：", "en": "翻譯成英文：", "ja": "翻譯成日文："}
DIRECTIONS = [("en", "zhtw"), ("zhtw", "en"), ("en", "ja"),
              ("ja", "en"), ("ja", "zhtw"), ("zhtw", "ja")]
WMT24PP = {"zhtw": "en-zh_TW", "ja": "en-ja_JP"}

# s2tw 而非 s2t：s2t 把「剛才→剛纔」「人群→人羣」等正確台灣用字誤判為簡體
cc_s2tw = OpenCC("s2tw")


# ---------------------------------------------------------------- 軸 A：文件級翻譯

def load_docs(n_docs):
    """WMT24++ en→zh_TW / en→ja 共用同一批英文原文 → 按 (document_id, segment_id)
    對齊即得六方向多平行文件。回傳 {(src,tgt): [(src_doc, ref_doc), ...]}。"""
    from datasets import load_dataset

    by_lang = {}
    for lang, config in WMT24PP.items():
        ds = load_dataset("google/wmt24pp", config, split="train")
        by_lang[lang] = {(r["document_id"], r["segment_id"]): r for r in ds
                         if not r["is_bad_source"]}
    keys = sorted(set(by_lang["zhtw"]) & set(by_lang["ja"]),
                  key=lambda k: (k[0], k[1]))
    assert keys, "wmt24pp zh_TW / ja 對不齊"

    docs = {}  # document_id -> {lang: [segments]}
    for doc_id, seg_id in keys:
        d = docs.setdefault(doc_id, {"en": [], "zhtw": [], "ja": []})
        d["en"].append(by_lang["zhtw"][(doc_id, seg_id)]["source"])
        d["zhtw"].append(by_lang["zhtw"][(doc_id, seg_id)]["target"])
        d["ja"].append(by_lang["ja"][(doc_id, seg_id)]["target"])
    # 只取多句文件（單句文件測不出長篇能力）
    picked = [d for d in docs.values() if len(d["en"]) >= 3][:n_docs]
    print(f"  {len(picked)} 篇文件，平均 {sum(len(d['en']) for d in picked)/len(picked):.1f} 句")

    out = {}
    for src, tgt in DIRECTIONS:
        out[(src, tgt)] = [("\n".join(d[src]), "\n".join(d[tgt])) for d in picked]
    return out


def repeat_ratio(text, n=8):
    """字元 n-gram 重複率 — 抓「同一句無限複述」的退化模式。"""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(v - 1 for v in c.values()) / len(grams)


def score_doc(direction, srcs, hyps, refs):
    import sacrebleu

    tgt = direction[1]
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    # 完整度：譯文長度 / 參考譯文長度。<1 代表漏譯，>>1 代表重複膨脹
    ratios = [len(h) / max(len(r), 1) for h, r in zip(hyps, refs)]
    ratios.sort()
    trunc = sum(r < 0.5 for r in ratios) / len(ratios) * 100   # 腰斬視為漏譯
    bloat = sum(r > 2.0 for r in ratios) / len(ratios) * 100   # 膨脹視為重複
    rep = sum(repeat_ratio(h) > 0.3 for h in hyps) / len(hyps) * 100
    # 逐「行」而非逐「文件」算洩漏：文件級一篇十幾句，只要一個簡體字整篇就算洩漏，
    # 分母太粗會直接飽和到 100%，也無法跟單句基準的洩漏率比較
    lines = [x for h in hyps for x in h.split("\n") if x.strip()]
    leak = (sum(cc_s2tw.convert(x) != x for x in lines) / len(lines) * 100
            if tgt == "zhtw" and lines else None)
    return {"chrf++": round(chrf, 2),
            "completeness_median": round(ratios[len(ratios) // 2], 3),
            "truncated_pct": round(trunc, 1),
            "bloated_pct": round(bloat, 1),
            "repetitive_pct": round(rep, 1),
            "simplified_leak_pct": round(leak, 2) if leak is not None else None}


# ------------------------------------------------- 軸 B：可驗證指令跟隨（IFEval 作法）

def _sep_only(text, sep, n):
    """恰好 n 項、只用 sep 分隔、沒有多餘說明文字。"""
    t = text.strip().rstrip("。.")
    parts = [p for p in t.split(sep) if p.strip()]
    return len(parts) == n and "\n" not in t and len(t) < 40


IFEVAL = [
    # (語言, 指令, 判分函數)
    ("zhtw", "列出三種台灣常見的水果，只要名稱，用頓號分隔，不要任何其他文字。",
     lambda t: _sep_only(t, "、", 3)),
    ("zhtw", "用剛好兩句話描述下雨的傍晚，每句都要以「。」結尾。",
     lambda t: t.strip().count("。") == 2 and t.strip().endswith("。")),
    ("zhtw", "回答「台北」這兩個字，不要標點，不要其他任何文字。",
     lambda t: t.strip() == "台北" or t.strip() == "臺北"),
    ("zhtw", "寫一段關於咖啡的介紹，全文不可以出現「咖啡」這兩個字。",
     lambda t: "咖啡" not in t and len(t.strip()) > 20),
    ("zhtw", "用繁體中文回答：什麼是機器學習？回答不超過 50 個字。",
     lambda t: len(t.strip()) <= 50 and cc_s2tw.convert(t) == t and len(t.strip()) > 5),
    ("zhtw", "把下面三個詞按筆畫由少到多排序，用箭頭 -> 連接，不要其他文字：山、樹、木",
     lambda t: "->" in t and t.strip().replace(" ", "").count("->") == 2),

    ("ja", "日本の都道府県を3つ、読点（、）で区切って名前だけ答えてください。他の文字は不要です。",
     lambda t: _sep_only(t, "、", 3)),
    ("ja", "「はい」だけを答えてください。他の文字は一切書かないでください。",
     lambda t: t.strip() in ("はい", "はい。")),
    ("ja", "雨の夕方を、ちょうど2つの文で描写してください。各文は「。」で終わること。",
     lambda t: t.strip().count("。") == 2 and t.strip().endswith("。")),
    ("ja", "機械学習とは何ですか。日本語で、50文字以内で答えてください。",
     lambda t: len(t.strip()) <= 50 and len(t.strip()) > 5
     and re.search(r"[぀-ヿ]", t) is not None),
    ("ja", "コーヒーについて説明してください。ただし「コーヒー」という単語は使わないでください。",
     lambda t: "コーヒー" not in t and len(t.strip()) > 20),

    ("en", "List exactly three colors, separated by commas, with no other text.",
     lambda t: _sep_only(t, ",", 3)),
    ("en", "Answer with the single word YES in all capital letters. Nothing else.",
     lambda t: t.strip().rstrip(".") == "YES"),
    ("en", "Describe a rainy evening in exactly two sentences ending with periods.",
     lambda t: t.strip().count(".") == 2 and t.strip().endswith(".")),
    ("en", "Explain machine learning in under 30 words.",
     lambda t: 3 < len(t.split()) <= 30),
    ("en", "Write about tea without using the letter 'e' anywhere in your answer.",
     lambda t: "e" not in t.lower() and len(t.strip()) > 20),
    ("en", "Reply with valid JSON only: an object with keys \"city\" and \"country\" "
           "for Taipei. No markdown fences, no explanation.",
     lambda t: _is_json_with(t, {"city", "country"})),
]


def _is_json_with(text, keys):
    try:
        obj = json.loads(text.strip())
    except Exception:
        return False
    return isinstance(obj, dict) and keys <= set(obj)


# ------------------------------------------- 軸 C：通用能力保留 + 退化成翻譯機偵測

GENERAL = [
    # (問題, 判分函數, 提示語言) — 答案唯一、可字串比對
    ("台灣最高的山是哪一座？", lambda t: "玉山" in t, "zhtw"),
    ("一件衣服原價 1200 元，打七折後再折 100 元，最後要付多少元？只要數字。",
     lambda t: "740" in t, "zhtw"),
    ("如果今天是星期三，三天後是星期幾？", lambda t: "六" in t or "星期六" in t, "zhtw"),
    ("台灣的首都是哪裡？", lambda t: "台北" in t or "臺北" in t, "zhtw"),
    ("水在攝氏幾度會沸騰？（標準大氣壓）", lambda t: "100" in t, "zhtw"),
    ("日本の首都はどこですか。", lambda t: "東京" in t, "ja"),
    ("1年は何ヶ月ですか。", lambda t: "12" in t or "十二" in t, "ja"),
    ("富士山の高さは約何メートルですか。",
     lambda t: "3776" in t or "3,776" in t or "3776" in t.replace(" ", ""), "ja"),
    ("What is the capital of Japan?", lambda t: "Tokyo" in t, "en"),
    ("What is 17 multiplied by 3?", lambda t: "51" in t, "en"),
    ("Write a Python function that returns the sum of even numbers in a list.",
     lambda t: "def" in t and "%" in t and "2" in t, "en"),
    ("Name the largest planet in our solar system.", lambda t: "Jupiter" in t, "en"),
]


def is_mere_translation(prompt, answer):
    """偵測「模型把問題翻譯掉而不是回答」——v2/v3 的招牌失敗模式。

    判據：答案長度與問題相近（0.5~2.0 倍）、答案主要文字系統與問題不同、
    且答案結尾仍是問句。任一組合成立即視為翻譯機行為。"""
    p, a = prompt.strip(), answer.strip()
    ratio = len(a) / max(len(p), 1)
    if not (0.4 <= ratio <= 2.5):
        return False
    han = re.compile(r"[一-鿿]")
    kana = re.compile(r"[぀-ヿ]")
    latin = re.compile(r"[A-Za-z]")
    # 純數字/算式答案不算翻譯（答錯是答錯，不是翻譯機）
    if len(han.findall(a)) + len(kana.findall(a)) + len(latin.findall(a)) < 5:
        return False

    def sig(s):
        return (bool(kana.search(s)), bool(han.search(s)),
                len(latin.findall(s)) > len(s) * 0.3)
    switched = sig(p) != sig(a)
    still_question = a.rstrip().endswith(("？", "?"))
    return switched or still_question


# ------------------------------------------------------------------------ 推論

def stop_token_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(ids)


def generate(tok, model, convs, max_new, batch):
    eos = stop_token_ids(tok)
    out = []
    for i in range(0, len(convs), batch):
        inp = tok.apply_chat_template(convs[i:i + batch], add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt",
                                      padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id)
        n = inp["input_ids"].shape[1]
        out.extend(tok.decode(x[n:], skip_special_tokens=True).strip() for x in g)
        print(f"    {min(i + batch, len(convs))}/{len(convs)}", flush=True)
    return out


def run_doc(tok, model, n_docs, batch, hyp_dir):
    print("== 軸 A：文件級翻譯（WMT24++）==")
    pairs = load_docs(n_docs)
    results = {}
    for (src_l, tgt_l), rows in pairs.items():
        name = f"{src_l}->{tgt_l}"
        print(f"  [{name}] {len(rows)} 篇")
        srcs = [s for s, _ in rows]
        refs = [r for _, r in rows]
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{INSTR[tgt_l]}\n{s}"}] for s in srcs]
        hyps = generate(tok, model, convs, max_new=2048, batch=batch)
        results[name] = score_doc((src_l, tgt_l), srcs, hyps, refs)
        print(f"    {results[name]}")
        stem = name.replace("->", "2")
        for suf, rows_ in (("src", srcs), ("ref", refs), ("hyp", hyps)):
            (hyp_dir / f"{stem}.{suf}.txt").write_text(
                "\n<<<DOC>>>\n".join(rows_), encoding="utf-8", newline="\n")
    return results


def run_ifeval(tok, model, batch, hyp_dir):
    print("== 軸 B：可驗證指令跟隨 ==")
    convs = [[{"role": "user", "content": q}] for _, q, _ in IFEVAL]
    outs = generate(tok, model, convs, max_new=512, batch=batch)
    per_lang, detail = Counter(), []
    tot = Counter()
    for (lang, q, check), o in zip(IFEVAL, outs):
        try:
            ok = bool(check(o))
        except Exception:
            ok = False
        per_lang[lang] += ok
        tot[lang] += 1
        detail.append({"lang": lang, "instruction": q, "output": o, "pass": ok})
    (hyp_dir / "ifeval.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    res = {lg: round(per_lang[lg] / tot[lg] * 100, 1) for lg in tot}
    res["overall"] = round(sum(per_lang.values()) / sum(tot.values()) * 100, 1)
    print(f"  {res}")
    return res


def run_general(tok, model, batch, hyp_dir):
    print("== 軸 C：通用能力保留 ==")
    convs = [[{"role": "user", "content": q}] for q, _, _ in GENERAL]
    outs = generate(tok, model, convs, max_new=512, batch=batch)
    correct, mere = 0, 0
    detail = []
    for (q, check, lang), o in zip(GENERAL, outs):
        try:
            ok = bool(check(o))
        except Exception:
            ok = False
        mt = is_mere_translation(q, o)
        correct += ok
        mere += mt
        detail.append({"lang": lang, "question": q, "output": o,
                       "correct": ok, "mere_translation": mt})
    (hyp_dir / "general.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(GENERAL)
    res = {"accuracy_pct": round(correct / n * 100, 1),
           "mere_translation_pct": round(mere / n * 100, 1), "n": n}
    print(f"  {res}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--axis", default="all", choices=["all", "doc", "ifeval", "general"])
    ap.add_argument("--docs", type=int, default=25, help="文件級評測取幾篇")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    print("== loading model ==")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  adapter: {args.adapter}")
    model.eval()

    hyp_dir = ROOT / "results" / "capability" / args.tag
    hyp_dir.mkdir(parents=True, exist_ok=True)
    dest = ROOT / "results" / "capability" / f"{args.tag}.json"
    # 單軸執行時保留其他軸既有結果，不覆蓋
    out = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    out |= {"tag": args.tag, "model": args.model, "adapter": args.adapter}
    axes = ["doc", "ifeval", "general"] if args.axis == "all" else [args.axis]
    if "doc" in axes:
        out["doc"] = run_doc(tok, model, args.docs, args.batch, hyp_dir)
    if "ifeval" in axes:
        out["ifeval"] = run_ifeval(tok, model, args.batch, hyp_dir)
    if "general" in axes:
        out["general"] = run_general(tok, model, args.batch, hyp_dir)

    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults -> {dest}")


if __name__ == "__main__":
    main()
