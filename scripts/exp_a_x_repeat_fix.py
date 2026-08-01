"""A 方案消融：在 x_repeat 上试 DECODE / en-nrng / 分段，看能否不训就止血。

只跑 v5e + 梁文锋案（已知会 LOOP）+ Grok Build 作对照。
"""
from __future__ import annotations

import gc
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

from repro_x_repeat import loop_stats  # same dir scripts/

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "data" / "manual_tests" / "x_repeat"
BASE = "Qwen/Qwen3.5-0.8B"
ADAPTER = ROOT / "outputs" / "sft-v5e"
INSTR = {
    "zhtw": "翻譯成繁體中文：",
    "en": "Translate to English:",
    "ja": "翻譯成日文：",
}
SHIP = {
    "ja": {"repetition_penalty": 1.1, "no_repeat_ngram_size": 4},
    "en": {"repetition_penalty": 1.1},
    "zhtw": {"no_repeat_ngram_size": 4},
}


def load_v5e():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model = PeftModel.from_pretrained(model, str(ADAPTER)).eval()
    eos = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    return tok, model, eos


def gen_one(tok, model, eos, text, tgt, max_new, gen_extra, beams=4):
    convs = [
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": f"{INSTR[tgt]}\n{text}"},
    ]
    inputs = tok.apply_chat_template(
        convs, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    kw = {
        "max_new_tokens": max_new,
        "do_sample": False,
        "num_beams": beams,
        "eos_token_id": eos,
        "pad_token_id": tok.pad_token_id,
        **gen_extra,
    }
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    return tok.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def split_chunks(text: str, max_chars: int = 280) -> list[str]:
    """按空行/句读切，单块不超过 max_chars（尽量不切断句子）。"""
    paras = re.split(r"\n\s*\n", text.strip())
    units: list[str] = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            units.append(p)
            continue
        # 中日文句读 / 英文 .!?
        sents = re.split(r"(?<=[。！？.!?\n])\s*", p)
        buf = ""
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf = f"{buf} {s}" if buf[-1] not in "。！？.!?\n" else buf + s
            else:
                units.append(buf)
                buf = s
        if buf:
            units.append(buf)
    # 仍超长则硬切
    out: list[str] = []
    for u in units:
        if len(u) <= max_chars * 2:
            out.append(u)
        else:
            for i in range(0, len(u), max_chars):
                out.append(u[i : i + max_chars])
    return out


def translate_chunked(tok, model, eos, text, tgt, max_new_per, gen_extra, beams=4):
    chunks = split_chunks(text, max_chars=280)
    hyps = []
    for i, ch in enumerate(chunks):
        # 每块给够空间，避免块内再爆
        mn = min(max_new_per, max(128, len(ch) * 3))
        hyps.append(gen_one(tok, model, eos, ch, tgt, mn, gen_extra, beams=beams))
    return "\n".join(hyps), len(chunks)


def main():
    man = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in man["cases"]}
    focus = [
        ("x_falali_liang_wefeng_investors", "en"),
        ("x_falali_liang_wefeng_investors", "ja"),
        ("x_xfreeze_grok_build", "zhtw"),
        ("x_xfreeze_grok_build", "ja"),
    ]

    strategies = [
        ("ship", lambda t: {**SHIP[t]}, 4, False),
        ("greedy", lambda t: {}, 1, False),
        ("ship_en_nrng3", lambda t: {**SHIP[t], **({"no_repeat_ngram_size": 3} if t == "en" else {})}, 4, False),
        ("ship_en_nrng4", lambda t: {**SHIP[t], **({"no_repeat_ngram_size": 4} if t == "en" else {})}, 4, False),
        ("ship_en_nrng6", lambda t: {**SHIP[t], **({"no_repeat_ngram_size": 6} if t == "en" else {})}, 4, False),
        ("ship_chunk280", lambda t: {**SHIP[t]}, 4, True),
        ("ship_en_nrng4_chunk280", lambda t: {**SHIP[t], **({"no_repeat_ngram_size": 4} if t == "en" else {})}, 4, True),
        ("greedy_chunk280", lambda t: {}, 1, True),
    ]

    print("loading v5e...", flush=True)
    tok, model, eos = load_v5e()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = FIXTURE / "runs" / f"expA_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid, tgt in focus:
        c = cases[sid]
        src = (FIXTURE / c["source_file"]).read_text(encoding="utf-8")
        for name, extra_fn, beams, chunked in strategies:
            # en-nrng 只对目标=en 有意义；其它目标会与 ship/chunk 重复
            if "en_nrng" in name and tgt != "en":
                continue
            extra = extra_fn(tgt)
            t0 = time.time()
            if chunked:
                hyp, nch = translate_chunked(
                    tok, model, eos, src, tgt, max_new_per=384, gen_extra=extra, beams=beams
                )
            else:
                # 全长：给 768 让循环有机会出现
                hyp = gen_one(tok, model, eos, src, tgt, max_new=768, gen_extra=extra, beams=beams)
                nch = 1
            dt = time.time() - t0
            st = loop_stats(hyp)
            row = {
                "strategy": name,
                "case": sid,
                "tgt": tgt,
                "src_chars": len(src),
                "n_chunks": nch,
                "sec": round(dt, 1),
                **st,
                "hyp_preview": hyp[:240],
                "hyp_tail": hyp[-200:],
            }
            rows.append(row)
            tag = f"{name}__{sid}__{tgt}"
            (out_dir / f"{tag}.hyp.txt").write_text(hyp, encoding="utf-8")
            flag = "LOOP" if st["loop_flag"] else "ok"
            print(
                f"[{flag:4}] {name:24} {sid[:22]:22} ->{tgt:4} "
                f"sent×{st['max_sentence_count']} w20×{st['max_window20_count']} "
                f"chars={st['chars']} {dt:.0f}s",
                flush=True,
            )

    (out_dir / "expA_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 判定 A 是否过关：梁文锋 en/ja 必须无 LOOP
    critical = [r for r in rows if r["case"].startswith("x_falali") and r["tgt"] in ("en", "ja")]
    by_s: dict[str, list] = {}
    for r in critical:
        by_s.setdefault(r["strategy"], []).append(r)
    print("\n=== A 方案门槛：梁文锋 en+ja 皆无 LOOP ===")
    winners = []
    for name, rs in by_s.items():
        ok = all(not r["loop_flag"] for r in rs)
        print(f"  {name:24} {'PASS' if ok else 'FAIL'}  " +
              " ".join(f"{r['tgt']}:{'L' if r['loop_flag'] else 'ok'}" for r in rs))
        if ok:
            winners.append(name)

    verdict = {
        "winners": winners,
        "A_ok": bool(winners),
        "recommend": winners[0] if winners else None,
        "next": "adopt_inference_sop" if winners else "go_to_B_training",
    }
    (out_dir / "verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nVERDICT", json.dumps(verdict, ensure_ascii=False))
    print("saved", out_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
