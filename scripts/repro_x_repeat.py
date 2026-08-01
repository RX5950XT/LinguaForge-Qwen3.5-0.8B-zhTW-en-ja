"""复现 data/manual_tests/x_repeat 的句级无限重复（base vs v5e）。

用法：
  uv run python scripts/repro_x_repeat.py
  uv run python scripts/repro_x_repeat.py --case x_xfreeze_grok_build --model v5e
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "data" / "manual_tests" / "x_repeat"
BASE = "Qwen/Qwen3.5-0.8B"
ADAPTER = ROOT / "outputs" / "sft-v5e"

INSTR = {
    "zhtw": "翻譯成繁體中文：",
    "en": "Translate to English:",
    "ja": "翻譯成日文：",
}
# 与 evaluate.DECODE 对齐（F57 后 en 也开 no_repeat_ngram=4）
DECODE = {
    "ja": {"repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    "en": {"repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    "zhtw": {"no_repeat_ngram_size": 4},
}


def loop_stats(text: str, n: int = 12) -> dict:
    """检测句级/短语级刷屏（不只滑动 n-gram 的相邻相同）。"""
    if len(text) < n:
        return {
            "max_phrase_count": 0,
            "worst_phrase": "",
            "unique_line_ratio": 1.0,
            "loop_flag": False,
            "chars": len(text),
            "regex_multi_repeat": 0,
        }
    # 1) 短句重复（按句读/换行切）
    parts = re.split(r"(?<=[。．.!?\n])\s*", text)
    parts = [x.strip() for x in parts if len(x.strip()) >= 8]
    sent_c = Counter(parts)
    worst_s, max_s = (sent_c.most_common(1)[0] if sent_c else ("", 0))
    # 2) 任意 16~40 字窗口总出现次数（抓 "it's gone, it's gone"）
    win_c: Counter[str] = Counter()
    w = 20
    for i in range(max(0, len(text) - w + 1)):
        win_c[text[i : i + w]] += 1
    worst_w, max_w = (win_c.most_common(1)[0] if win_c else ("", 0))
    # 3) 紧挨着的多重复 (...X X X...)
    multi = re.findall(r"(.{12,80}?)\1{3,}", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    uniq = len(set(lines)) / max(len(lines), 1)
    max_phrase = max(int(max_s), int(max_w))
    worst = worst_s if max_s >= max_w else worst_w
    loop = (
        max_s >= 4
        or max_w >= 8
        or bool(multi)
        or (max_s >= 3 and len(worst_s) >= 12)
    )
    return {
        "max_phrase_count": max_phrase,
        "worst_phrase": worst[:80],
        "unique_line_ratio": round(uniq, 3),
        "loop_flag": bool(loop),
        "chars": len(text),
        "max_sentence_count": int(max_s),
        "max_window20_count": int(max_w),
        "regex_multi_repeat": len(multi),
    }


def load_model(which: str):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    if which == "v5e":
        model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    eos = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    return tok, model, eos


def translate(
    tok,
    model,
    eos,
    text: str,
    tgt: str,
    max_new: int,
    *,
    decode_mode: str = "ship",
    beams: int | None = None,
) -> str:
    convs = [
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": f"{INSTR[tgt]}\n{text}"},
    ]
    inputs = tok.apply_chat_template(
        convs, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    gen_kwargs: dict = {
        "max_new_tokens": max_new,
        "do_sample": False,
        "eos_token_id": eos,
        "pad_token_id": tok.pad_token_id,
    }
    if decode_mode == "ship":
        gen_kwargs["num_beams"] = beams if beams is not None else 4
        gen_kwargs.update(DECODE[tgt])
    else:
        # 裸 greedy：贴近 llama-cli / 未设 DECODE 的调用；最容易爆循环
        gen_kwargs["num_beams"] = beams if beams is not None else 1
    with torch.no_grad():
        gen = model.generate(**inputs, **gen_kwargs)
    return tok.decode(
        gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="manifest case id；默认全部")
    ap.add_argument("--model", choices=["base", "v5e", "both"], default="both")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument(
        "--decode",
        choices=["ship", "greedy"],
        default="ship",
        help="ship=evaluate.DECODE（beam4+nrng）；greedy=无 beam/无防重复（贴 GGUF/裸 generate）",
    )
    ap.add_argument("--beams", type=int, default=None, help="覆盖 num_beams")
    args = ap.parse_args()

    man = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    cases = man["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            raise SystemExit(f"case not found: {args.case}")

    models = ["base", "v5e"] if args.model == "both" else [args.model]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = FIXTURE / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for which in models:
        print(f"\n=== load {which} ===", flush=True)
        tok, model, eos = load_model(which)
        for c in cases:
            src = (FIXTURE / c["source_file"]).read_text(encoding="utf-8")
            for tgt in c["targets"]:
                print(f"  {which} {c['id']} -> {tgt} ...", flush=True)
                hyp = translate(
                    tok,
                    model,
                    eos,
                    src,
                    tgt,
                    args.max_new,
                    decode_mode=args.decode,
                    beams=args.beams,
                )
                st = loop_stats(hyp)
                row = {
                    "model": which,
                    "case": c["id"],
                    "src_lang": c["src_lang"],
                    "tgt": tgt,
                    "decode": args.decode,
                    "src_chars": len(src),
                    **st,
                    "hyp_preview": hyp[:400],
                }
                report.append(row)
                tag = f"{which}__{c['id']}__{tgt}"
                (out_dir / f"{tag}.hyp.txt").write_text(hyp, encoding="utf-8")
                flag = "FAIL-LOOP" if st["loop_flag"] else "ok"
                print(
                    f"    {flag} chars={st['chars']} "
                    f"sent×{st.get('max_sentence_count', 0)} "
                    f"w20×{st.get('max_window20_count', 0)} "
                    f"phrase={st.get('worst_phrase', '')!r}",
                    flush=True,
                )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {out_dir}")
    # summary table
    print("\nSUMMARY")
    for r in report:
        print(
            f"{r['model']:4} {r['case'][:28]:28} ->{r['tgt']:4} "
            f"{'LOOP' if r['loop_flag'] else 'ok  ':4} "
            f"sent×{r.get('max_sentence_count', 0)} "
            f"w20×{r.get('max_window20_count', 0)} chars={r['chars']}"
        )


if __name__ == "__main__":
    main()
