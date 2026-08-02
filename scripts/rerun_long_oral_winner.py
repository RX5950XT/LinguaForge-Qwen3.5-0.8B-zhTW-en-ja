"""Re-run long-oral ZH→EN for ship winner with full kwargs (incl. length_penalty).

Usage:
  uv run python scripts/rerun_long_oral_winner.py [--scratch DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_search import (  # noqa: E402
    ADAPTER_DEFAULT,
    OUT_DIR,
    load_model,
    long_oral_loop_flag,
    write_json,
)
from evaluate import DECODE, LENGTH_PENALTY, NUM_BEAMS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--adapter", type=Path, default=ADAPTER_DEFAULT)
    args = ap.parse_args()

    lo_scratch = None
    if args.scratch:
        lo_scratch = args.scratch / "long_oral"
        lo_scratch.mkdir(parents=True, exist_ok=True)

    print("loading model...", flush=True)
    tok, model = load_model(args.adapter)
    decode_en = DECODE["en"]
    results: dict = {}

    for name, lp in (("winner_lp1.2", LENGTH_PENALTY), ("control_lp1.0", 1.0)):
        print(f"long-oral {name} beams={NUM_BEAMS} lp={lp}", flush=True)
        st = long_oral_loop_flag(
            tok,
            model,
            decode_en,
            beams=NUM_BEAMS,
            max_new=768,
            length_penalty=lp,
        )
        hyp = st.pop("hyp")
        digest = hashlib.sha256(hyp.encode("utf-8")).hexdigest()
        rec = {
            **st,
            "sha256": digest,
            "length_penalty": lp,
            "beams": NUM_BEAMS,
            "decode_en": decode_en,
            "hyp_chars": len(hyp),
        }
        results[name] = rec
        hyp_name = f"{name}-long_oral.en.hyp.txt"
        stats_name = f"{name}-long_oral.stats.json"
        (OUT_DIR / hyp_name).write_text(hyp + "\n", encoding="utf-8")
        write_json(OUT_DIR / stats_name, rec)
        if lo_scratch is not None:
            (lo_scratch / f"{name}.en.hyp.txt").write_text(hyp + "\n", encoding="utf-8")
            write_json(lo_scratch / f"{name}.stats.json", rec)
        print(
            f"  loop_flag={rec['loop_flag']} chars={rec['hyp_chars']} "
            f"sha256={digest[:16]}... gen={rec['gen_kwargs']}",
            flush=True,
        )

    assert results["winner_lp1.2"]["loop_flag"] is False, "winner must not loop"
    assert results["winner_lp1.2"]["gen_kwargs"].get("length_penalty") == LENGTH_PENALTY
    assert results["control_lp1.0"]["gen_kwargs"].get("length_penalty") == 1.0

    same = results["winner_lp1.2"]["sha256"] == results["control_lp1.0"]["sha256"]
    summary = {
        "winner_loop_flag": results["winner_lp1.2"]["loop_flag"],
        "winner_sha256": results["winner_lp1.2"]["sha256"],
        "control_sha256": results["control_lp1.0"]["sha256"],
        "hyps_identical": same,
        "winner_gen_kwargs": results["winner_lp1.2"]["gen_kwargs"],
        "control_gen_kwargs": results["control_lp1.0"]["gen_kwargs"],
        "fix": "long_oral_loop_flag now passes length_penalty",
    }
    write_json(OUT_DIR / "winner_true_lp_verify.json", summary)
    if args.scratch:
        write_json(args.scratch / "long_oral" / "winner_true_lp_verify.json", summary)
        write_json(args.scratch / "winner_true_lp_verify.json", summary)

    # Canonical winner artifact names
    w_hyp = (OUT_DIR / "winner_lp1.2-long_oral.en.hyp.txt").read_text(encoding="utf-8")
    (OUT_DIR / "b4_lp1.2-long_oral.en.hyp.txt").write_text(w_hyp, encoding="utf-8")
    write_json(OUT_DIR / "b4_lp1.2-long_oral.stats.json", results["winner_lp1.2"])
    if lo_scratch is not None:
        (lo_scratch / "b4_lp1.2.en.hyp.txt").write_text(w_hyp, encoding="utf-8")
        write_json(lo_scratch / "b4_lp1.2.stats.json", results["winner_lp1.2"])

    cell_p = OUT_DIR / "b4_lp1.2-flores.json"
    if cell_p.exists():
        cell = json.loads(cell_p.read_text(encoding="utf-8"))
        cell["loop_flag_en"] = results["winner_lp1.2"]["loop_flag"]
        cell["long_oral"] = dict(results["winner_lp1.2"])
        cell["long_oral_note"] = (
            "re-captured with length_penalty=1.2 after long_oral_loop_flag fix"
        )
        write_json(cell_p, cell)
        if args.scratch:
            write_json(args.scratch / "b4_lp1.2-flores.json", cell)

    print(json.dumps(summary, indent=2), flush=True)
    print("LONG_ORAL_OK", flush=True)


if __name__ == "__main__":
    main()
