"""Decode-config search for ship model v5e (no retrain).

Uses the same generate + score path as evaluate.py (batched_translate / score /
load_flores). Pure ranking lives in rank_candidates() so it is unit-testable
without GPU.

Usage:
  uv run python scripts/decode_search.py --limit 200 --batch 16
  uv run python scripts/decode_search.py --limit 200 --only baseline_ship,b4_nrng6
  uv run python scripts/decode_search.py --rank-only results/decode_search/*.json

Outputs:
  results/decode_search/<id>-flores.json   (metrics per candidate)
  results/decode_search/ranking.json       (winner + constraints)
  Optional --scratch DIR also mirrors JSON + report there.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    DECODE,
    DIRECTIONS,
    INSTR,
    MODEL_ID,
    ROOT,
    batched_translate,
    load_flores,
    score,
)
from repro_x_repeat import loop_stats  # noqa: E402

ADAPTER_DEFAULT = ROOT / "outputs" / "sft-v5e"
OUT_DIR = ROOT / "results" / "decode_search"
LONG_ORAL = (
    ROOT / "data" / "manual_tests" / "x_repeat" / "sources" / "2080141498658762797.zh.txt"
)

# Leak hard gate: same spirit as CLAUDE.md (≤ base+0.3 vs *baseline of this search*).
LEAK_TOL = 0.3
# Soft quality: any direction chrF++ may not fall more than this below baseline.
DIR_DROP_TOL = 1.5


def baseline_decode() -> dict:
    """Today's shipped evaluate.DECODE (deep copy)."""
    return copy.deepcopy(DECODE)


def baseline_beams() -> int:
    from evaluate import NUM_BEAMS
    return NUM_BEAMS


def baseline_length_penalty() -> float:
    from evaluate import LENGTH_PENALTY
    return LENGTH_PENALTY


def _d(ja=None, en=None, zhtw=None) -> dict:
    """Build per-target decode maps; None keys inherit baseline."""
    b = baseline_decode()
    out = {}
    for lang, override in (("ja", ja), ("en", en), ("zhtw", zhtw)):
        out[lang] = dict(b[lang] if override is None else override)
    return out


def search_space() -> list[dict]:
    """Named coarse grid. First entry is the named baseline (ship defaults)."""
    b = baseline_decode()
    lp = baseline_length_penalty()
    return [
        {
            "id": "baseline_ship",
            "beams": baseline_beams(),
            "decode": b,
            "length_penalty": lp,
            "notes": "ship evaluate.DECODE + NUM_BEAMS + LENGTH_PENALTY",
        },
        {
            "id": "greedy_ship_decode",
            "beams": 1,
            "decode": copy.deepcopy(b),
            "length_penalty": None,
            "notes": "same per-lang DECODE, greedy",
        },
        {
            "id": "b4_nrng0",
            "beams": 4,
            "decode": _d(
                ja={"repetition_penalty": 1.1},
                en={"repetition_penalty": 1.1},
                zhtw={},
            ),
            "length_penalty": lp,
            "notes": "beam4, no no_repeat_ngram (pre-F11 style tails risk)",
        },
        {
            "id": "b4_en_nrng0",
            "beams": 4,
            "decode": _d(
                en={"repetition_penalty": 1.1},  # F11: en without nrng
            ),
            "length_penalty": lp,
            "notes": "historical en without nrng (F57 loop risk)",
        },
        {
            "id": "b4_all_nrng3",
            "beams": 4,
            "decode": _d(
                ja={"repetition_penalty": 1.1, "no_repeat_ngram_size": 3},
                en={"repetition_penalty": 1.1, "no_repeat_ngram_size": 3},
                zhtw={"no_repeat_ngram_size": 3},
            ),
            "length_penalty": lp,
            "notes": "nrng=3 all targets",
        },
        {
            "id": "b4_all_nrng6",
            "beams": 4,
            "decode": _d(
                ja={"repetition_penalty": 1.1, "no_repeat_ngram_size": 6},
                en={"repetition_penalty": 1.1, "no_repeat_ngram_size": 6},
                zhtw={"no_repeat_ngram_size": 6},
            ),
            "length_penalty": lp,
            "notes": "nrng=6 all targets",
        },
        {
            "id": "b4_rep1.0",
            "beams": 4,
            "decode": _d(
                ja={"no_repeat_ngram_size": 4},
                en={"no_repeat_ngram_size": 4},
                zhtw={"no_repeat_ngram_size": 4},
            ),
            "length_penalty": lp,
            "notes": "nrng=4, no repetition_penalty",
        },
        {
            "id": "b4_en_nrng3",
            "beams": 4,
            "decode": _d(
                en={"repetition_penalty": 1.1, "no_repeat_ngram_size": 3},
            ),
            "length_penalty": lp,
            "notes": "ship but en nrng=3",
        },
        {
            "id": "b4_lp1.0",
            "beams": 4,
            "decode": copy.deepcopy(b),
            "length_penalty": 1.0,
            "notes": "ship DECODE but length_penalty 1.0 (pre-2026-08-02)",
        },
        {
            "id": "b2_ship_decode",
            "beams": 2,
            "decode": copy.deepcopy(b),
            "length_penalty": lp,
            "notes": "beams=2 intermediate",
        },
    ]


def avg_chrf(results: dict) -> float:
    vals = [r["chrf++"] for r in results.values() if r.get("chrf++") is not None]
    return sum(vals) / len(vals) if vals else float("-inf")


def leak_map(results: dict) -> dict[str, float | None]:
    return {
        k: r.get("simplified_leak_pct")
        for k, r in results.items()
        if k.endswith("->zhtw")
    }


def rank_candidates(
    cells: list[dict],
    *,
    leak_tol: float = LEAK_TOL,
    dir_drop_tol: float = DIR_DROP_TOL,
    require_long_oral_ok: bool = True,
) -> dict:
    """Pure ranking. Each cell needs: id, results, optional loop_flag_en.

    Rule (written, multi-objective):
      1. Drop if long-oral ZH→EN loop_flag is True when require_long_oral_ok
         and the field is present.
      2. Drop if any zhtw-direction leak > baseline_leak + leak_tol.
      3. Drop if any direction chrF++ < baseline_dir - dir_drop_tol.
      4. Among remaining: maximize mean chrF++ (6 dirs).
      5. Tie-break: lower max zhtw leak, then prefer id == baseline_ship,
         then lexicographic id.
    """
    if not cells:
        return {"winner": None, "candidates": [], "rule": "empty"}

    by_id = {c["id"]: c for c in cells}
    if "baseline_ship" not in by_id:
        raise ValueError("cells must include baseline_ship")
    base = by_id["baseline_ship"]
    base_res = base["results"]
    base_avg = avg_chrf(base_res)
    base_leaks = leak_map(base_res)

    ranked = []
    for c in cells:
        res = c["results"]
        reasons_fail = []
        if require_long_oral_ok and "loop_flag_en" in c and c["loop_flag_en"]:
            reasons_fail.append("long_oral_loop")
        for d, leak in leak_map(res).items():
            if leak is None:
                continue
            b_leak = base_leaks.get(d)
            if b_leak is not None and leak > b_leak + leak_tol:
                reasons_fail.append(f"leak:{d}:{leak}>{b_leak}+{leak_tol}")
        for d, r in res.items():
            ch = r.get("chrf++")
            bch = base_res.get(d, {}).get("chrf++")
            if ch is not None and bch is not None and ch < bch - dir_drop_tol:
                reasons_fail.append(f"dir_drop:{d}:{ch}<{bch}-{dir_drop_tol}")
        mean = avg_chrf(res)
        max_leak = max(
            (v for v in leak_map(res).values() if v is not None), default=0.0
        )
        ok = not reasons_fail
        ranked.append(
            {
                "id": c["id"],
                "ok": ok,
                "fail_reasons": reasons_fail,
                "chrf_avg": round(mean, 4),
                "delta_vs_baseline": round(mean - base_avg, 4),
                "max_zhtw_leak": max_leak,
                "loop_flag_en": c.get("loop_flag_en"),
                "beams": c.get("beams"),
                "decode": c.get("decode"),
                "length_penalty": c.get("length_penalty"),
                "results": res,
            }
        )

    ok_rows = [r for r in ranked if r["ok"]]
    pool = ok_rows if ok_rows else []  # empty → no winner among constrained

    def sort_key(r):
        return (
            r["chrf_avg"],
            -r["max_zhtw_leak"],
            1 if r["id"] == "baseline_ship" else 0,
            r["id"],
        )

    ok_rows_sorted = sorted(pool, key=sort_key, reverse=True)
    all_sorted = sorted(ranked, key=lambda r: (r["ok"], r["chrf_avg"]), reverse=True)
    winner = ok_rows_sorted[0]["id"] if ok_rows_sorted else None
    return {
        "winner": winner,
        "baseline_id": "baseline_ship",
        "baseline_chrf_avg": round(base_avg, 4),
        "rule": {
            "primary": "max mean chrF++ over six directions",
            "hard": [
                f"long_oral ZH→EN loop_flag must be false when measured",
                f"zhtw leak ≤ baseline + {leak_tol}",
                f"no direction chrF++ < baseline − {dir_drop_tol}",
            ],
            "tie_break": "lower max zhtw leak; prefer baseline_ship; id",
        },
        "candidates": all_sorted,
        "eligible": [r["id"] for r in ok_rows_sorted],
        "winner_equals_baseline": winner == "baseline_ship",
    }


def run_candidate(
    tok,
    model,
    cand: dict,
    pairs: dict,
    batch: int,
    tag_prefix: str,
) -> dict:
    """Score one candidate on preloaded FLORES pairs via evaluate.batched_translate."""
    beams = cand["beams"]
    decode = cand["decode"]
    lp = cand.get("length_penalty")
    results = {}
    n = 0
    for (src_l, tgt_l), (src_sents, refs) in pairs.items():
        name = f"{src_l}->{tgt_l}"
        n = len(src_sents)
        print(f"  [{cand['id']}] {name} (n={n})", flush=True)
        prompts = [f"{INSTR[tgt_l]}\n{s}" for s in src_sents]
        gen_kwargs = dict(decode[tgt_l])
        if beams > 1:
            gen_kwargs["num_beams"] = beams
        if lp is not None:
            gen_kwargs["length_penalty"] = lp
        # Drop empty dict noise
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        hyps = batched_translate(tok, model, prompts, batch, gen_kwargs)
        results[name] = score((src_l, tgt_l), hyps, refs)
        print(f"    {results[name]}", flush=True)

    payload = {
        "id": cand["id"],
        "tag": f"{tag_prefix}-{cand['id']}",
        "model": MODEL_ID,
        "adapter": str(ADAPTER_DEFAULT),
        "n": n,
        "beams": beams,
        "decode": decode,
        "length_penalty": lp,
        "gen_shared": {
            "max_new_tokens": 256,
            "do_sample": False,
            "num_beams": beams if beams > 1 else 1,
            "length_penalty": lp,
        },
        "notes": cand.get("notes"),
        "results": results,
        "chrf_avg": round(avg_chrf(results), 4),
    }
    return payload


def build_long_oral_gen_kwargs(
    decode_en: dict,
    beams: int,
    length_penalty: float | None = None,
) -> dict:
    """Full winner/candidate kwargs for long-oral ZH→EN (pure; unit-testable).

    Must mirror FLORES path: decode[tgt] + num_beams + length_penalty.
    Omitting length_penalty would silently run HF default 1.0 and make
    b4_lp1.2 hyps identical to baseline_ship — that bug was fixed 2026-08-02.
    """
    gen_kwargs = {k: v for k, v in decode_en.items() if v is not None}
    if beams > 1:
        gen_kwargs["num_beams"] = beams
    if length_penalty is not None:
        gen_kwargs["length_penalty"] = length_penalty
    return gen_kwargs


def long_oral_loop_flag(
    tok,
    model,
    decode_en: dict,
    beams: int,
    max_new: int = 768,
    length_penalty: float | None = None,
) -> dict:
    """ZH→EN long-oral case; uses evaluate INSTR + stop tokens path.

    length_penalty is required for any candidate that sets it (e.g. ship 1.2).
    """
    from evaluate import stop_token_ids, SYSTEM

    text = LONG_ORAL.read_text(encoding="utf-8")
    convs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{INSTR['en']}\n{text}"},
    ]
    inputs = tok.apply_chat_template(
        convs, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    gen_kwargs = build_long_oral_gen_kwargs(decode_en, beams, length_penalty)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=stop_token_ids(tok),
            pad_token_id=tok.pad_token_id,
            **gen_kwargs,
        )
    n_in = inputs["input_ids"].shape[1]
    hyp = tok.decode(gen[0][n_in:], skip_special_tokens=True).strip()
    st = loop_stats(hyp)
    st["hyp"] = hyp
    st["gen_kwargs"] = gen_kwargs  # record what was actually used
    return st


def load_model(adapter: Path):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return tok, model


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200, help="FLORES slice size (same for all)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--adapter", type=Path, default=ADAPTER_DEFAULT)
    ap.add_argument("--only", type=str, default=None, help="comma-separated candidate ids")
    ap.add_argument("--skip-long-oral", action="store_true")
    ap.add_argument("--rank-only", nargs="*", help="rank existing cell JSON paths")
    ap.add_argument("--scratch", type=Path, default=None, help="mirror outputs here")
    ap.add_argument("--tag-prefix", default="v5e-decode")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.rank_only is not None:
        cells = []
        paths = args.rank_only or list(OUT_DIR.glob("*-flores.json"))
        for p in paths:
            p = Path(p)
            if p.name == "ranking.json":
                continue
            cells.append(json.loads(p.read_text(encoding="utf-8")))
        ranking = rank_candidates(cells)
        write_json(OUT_DIR / "ranking.json", ranking)
        if args.scratch:
            write_json(args.scratch / "ranking.json", ranking)
        print(json.dumps({"winner": ranking["winner"], "eligible": ranking["eligible"]}, indent=2))
        return

    space = search_space()
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        space = [c for c in space if c["id"] in want]
        if "baseline_ship" not in {c["id"] for c in space}:
            # always need baseline for ranking constraints when full rank later
            pass

    print("== search space ==")
    for c in space:
        print(f"  {c['id']}: beams={c['beams']} lp={c['length_penalty']} {c['notes']}")

    print(f"== loading model adapter={args.adapter} ==")
    tok, model = load_model(args.adapter)
    print(f"== loading FLORES limit={args.limit} ==")
    pairs = load_flores(args.limit)
    assert pairs, "no FLORES pairs"
    n_dirs = len(pairs)
    n_sents = len(next(iter(pairs.values()))[0])
    print(f"  directions={n_dirs} n={n_sents}")

    cells = []
    for cand in space:
        print(f"\n== candidate {cand['id']} ==", flush=True)
        payload = run_candidate(tok, model, cand, pairs, args.batch, args.tag_prefix)
        if not args.skip_long_oral:
            print(f"  [{cand['id']}] long-oral ZH→EN ...", flush=True)
            st = long_oral_loop_flag(
                tok,
                model,
                cand["decode"]["en"],
                cand["beams"],
                max_new=768,
                length_penalty=cand.get("length_penalty"),
            )
            payload["loop_flag_en"] = st["loop_flag"]
            payload["long_oral"] = {
                k: st[k] for k in st if k != "hyp"
            }
            payload["long_oral"]["hyp_chars"] = len(st["hyp"])
            payload["long_oral"]["length_penalty"] = cand.get("length_penalty")
            # keep hyp only under scratch / side file to keep metrics json smaller
            hyp_path = OUT_DIR / f"{cand['id']}-long_oral.en.hyp.txt"
            hyp_path.write_text(st["hyp"] + "\n", encoding="utf-8")
            print(
                f"    loop_flag={st['loop_flag']} chars={st['chars']} "
                f"max_phrase={st['max_phrase_count']}",
                flush=True,
            )
        out = OUT_DIR / f"{cand['id']}-flores.json"
        write_json(out, payload)
        if args.scratch:
            write_json(args.scratch / out.name, payload)
            if "long_oral" in payload:
                lo = args.scratch / "long_oral" / f"{cand['id']}.en.hyp.txt"
                lo.parent.mkdir(parents=True, exist_ok=True)
                lo.write_text(
                    (OUT_DIR / f"{cand['id']}-long_oral.en.hyp.txt").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                write_json(args.scratch / "long_oral" / f"{cand['id']}.stats.json", payload["long_oral"])
        cells.append(payload)
        print(f"  chrf_avg={payload['chrf_avg']} -> {out}")

    ranking = rank_candidates(cells)
    write_json(OUT_DIR / "ranking.json", ranking)
    if args.scratch:
        write_json(args.scratch / "ranking.json", ranking)
        # human table
        lines = [
            "# Decode search ranking (v5e)",
            "",
            f"n={n_sents} FLORES slice; model={MODEL_ID}; adapter={args.adapter}",
            f"winner={ranking['winner']} equals_baseline={ranking['winner_equals_baseline']}",
            "",
            "| id | ok | chrf_avg | Δbase | max_leak | loop_en | beams |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in ranking["candidates"]:
            lines.append(
                f"| {r['id']} | {r['ok']} | {r['chrf_avg']} | {r['delta_vs_baseline']} | "
                f"{r['max_zhtw_leak']} | {r['loop_flag_en']} | {r['beams']} |"
            )
        lines.append("")
        lines.append("## Rule")
        lines.append(json.dumps(ranking["rule"], ensure_ascii=False, indent=2))
        (args.scratch / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n== ranking ==")
    print(f"winner: {ranking['winner']}")
    print(f"eligible: {ranking['eligible']}")
    print(f"winner_equals_baseline: {ranking['winner_equals_baseline']}")


if __name__ == "__main__":
    main()
