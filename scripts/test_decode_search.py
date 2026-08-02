"""Unit tests for decode_search ranking (no GPU)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decode_search import (  # noqa: E402
    baseline_beams,
    baseline_decode,
    baseline_length_penalty,
    build_long_oral_gen_kwargs,
    rank_candidates,
    search_space,
)
from evaluate import DECODE, LENGTH_PENALTY, NUM_BEAMS  # noqa: E402


def _cell(cid, chrf_by_dir, leak_en=1.0, leak_ja=0.7, loop=False, **extra):
    results = {}
    for d, ch in chrf_by_dir.items():
        results[d] = {
            "chrf++": ch,
            "bleu": ch * 0.5,
            "simplified_leak_pct": (
                leak_en if d == "en->zhtw" else leak_ja if d == "ja->zhtw" else None
            ),
        }
    return {
        "id": cid,
        "results": results,
        "loop_flag_en": loop,
        "beams": 4,
        "decode": baseline_decode(),
        **extra,
    }


DIRS = [
    "en->zhtw",
    "zhtw->en",
    "en->ja",
    "ja->en",
    "ja->zhtw",
    "zhtw->ja",
]


def _flat(v):
    return {d: v for d in DIRS}


def test_baseline_matches_ship_decode():
    b = baseline_decode()
    assert b == DECODE
    assert baseline_beams() == NUM_BEAMS == 4
    assert baseline_length_penalty() == LENGTH_PENALTY == 1.2
    space = search_space()
    assert space[0]["id"] == "baseline_ship"
    assert space[0]["decode"] == DECODE
    assert space[0]["beams"] == NUM_BEAMS
    assert space[0]["length_penalty"] == LENGTH_PENALTY
    ids = {c["id"] for c in space}
    assert "b4_en_nrng0" in ids and "b4_all_nrng6" in ids


def test_winner_is_baseline_when_tied():
    base = _cell("baseline_ship", _flat(30.0))
    other = _cell("other", _flat(30.0))
    r = rank_candidates([base, other])
    assert r["winner"] == "baseline_ship"
    assert r["winner_equals_baseline"] is True


def test_higher_chrf_wins_if_constraints_ok():
    base = _cell("baseline_ship", _flat(30.0), leak_en=1.0)
    better = _cell("better", _flat(31.0), leak_en=1.0)
    r = rank_candidates([base, better])
    assert r["winner"] == "better"


def test_loop_disqualifies():
    base = _cell("baseline_ship", _flat(30.0), loop=False)
    loopy = _cell("loopy", _flat(40.0), loop=True)
    r = rank_candidates([base, loopy])
    assert r["winner"] == "baseline_ship"
    assert "loopy" not in r["eligible"]
    loopy_row = next(c for c in r["candidates"] if c["id"] == "loopy")
    assert loopy_row["ok"] is False
    assert "long_oral_loop" in loopy_row["fail_reasons"]


def test_leak_tolerance():
    base = _cell("baseline_ship", _flat(30.0), leak_en=1.0)
    bad = _cell("leaky", _flat(35.0), leak_en=1.0 + 0.3 + 0.01)
    ok = _cell("okish", _flat(30.5), leak_en=1.0 + 0.3)
    r = rank_candidates([base, bad, ok])
    assert "leaky" not in r["eligible"]
    assert r["winner"] == "okish"


def test_dir_drop_disqualifies():
    base_scores = _flat(30.0)
    drop = copy.deepcopy(base_scores)
    drop["zhtw->en"] = 28.0  # −2.0 > 1.5 tol
    base = _cell("baseline_ship", base_scores)
    bad = _cell("dropper", drop)
    r = rank_candidates([base, bad])
    assert "dropper" not in r["eligible"]
    assert r["winner"] == "baseline_ship"


def test_empty_eligible_when_all_fail_except_missing_baseline_loop():
    # baseline itself loops → no eligible winner
    base = _cell("baseline_ship", _flat(30.0), loop=True)
    r = rank_candidates([base])
    assert r["winner"] is None
    assert r["eligible"] == []


def test_long_oral_gen_kwargs_include_length_penalty():
    """Regression: long-oral must not drop length_penalty (skeptic 2026-08-02)."""
    en = DECODE["en"]
    # Winner path
    kw = build_long_oral_gen_kwargs(en, beams=4, length_penalty=LENGTH_PENALTY)
    assert kw["num_beams"] == 4
    assert kw["length_penalty"] == 1.2
    assert kw["no_repeat_ngram_size"] == 4
    assert kw["repetition_penalty"] == 1.1
    # Explicit omit → no key (HF default 1.0) — only when caller passes None
    kw0 = build_long_oral_gen_kwargs(en, beams=4, length_penalty=None)
    assert "length_penalty" not in kw0
    # Explicit 1.0 still recorded
    kw1 = build_long_oral_gen_kwargs(en, beams=4, length_penalty=1.0)
    assert kw1["length_penalty"] == 1.0
    # Ship baseline candidate from search_space carries lp into this builder
    base = next(c for c in search_space() if c["id"] == "baseline_ship")
    kw_ship = build_long_oral_gen_kwargs(
        base["decode"]["en"], base["beams"], base["length_penalty"]
    )
    assert kw_ship["length_penalty"] == LENGTH_PENALTY


def main():
    test_baseline_matches_ship_decode()
    test_winner_is_baseline_when_tied()
    test_higher_chrf_wins_if_constraints_ok()
    test_loop_disqualifies()
    test_leak_tolerance()
    test_dir_drop_disqualifies()
    test_empty_eligible_when_all_fail_except_missing_baseline_loop()
    test_long_oral_gen_kwargs_include_length_penalty()
    print("OK: test_decode_search (8 cases)")


if __name__ == "__main__":
    main()
