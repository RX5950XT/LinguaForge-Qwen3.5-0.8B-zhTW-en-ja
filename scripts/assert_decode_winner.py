"""Assert shipped evaluate decode defaults match decode_search winner kwargs.

Usage:
  uv run python scripts/assert_decode_winner.py results/decode_search/ranking.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import DECODE, LENGTH_PENALTY, NUM_BEAMS  # noqa: E402


def main(ranking_path: Path) -> None:
    r = json.loads(ranking_path.read_text(encoding="utf-8"))
    winner = r["winner"]
    assert winner, "no winner in ranking"
    wrow = next(c for c in r["candidates"] if c["id"] == winner)
    w_decode = wrow.get("decode")
    if w_decode is None:
        cell = json.loads(
            (ranking_path.parent / f"{winner}-flores.json").read_text(encoding="utf-8")
        )
        w_decode = cell["decode"]
        w_beams = cell["beams"]
        w_lp = cell.get("length_penalty")
    else:
        w_beams = wrow.get("beams", 4)
        w_lp = wrow.get("length_penalty")

    print("winner", winner)
    print("winner_decode", json.dumps(w_decode, ensure_ascii=False))
    print("winner_beams", w_beams, "winner_lp", w_lp)
    print("shipped_DECODE", json.dumps(DECODE, ensure_ascii=False))
    print("shipped_NUM_BEAMS", NUM_BEAMS, "shipped_LENGTH_PENALTY", LENGTH_PENALTY)

    assert DECODE == w_decode, (DECODE, w_decode)
    assert NUM_BEAMS == w_beams, (NUM_BEAMS, w_beams)
    # HF default length_penalty is 1.0 when omitted; ship maps that to LENGTH_PENALTY.
    expect_lp = 1.0 if w_lp is None else w_lp
    assert LENGTH_PENALTY == expect_lp, (LENGTH_PENALTY, expect_lp)
    print("ASSERT_OK: shipped DECODE/NUM_BEAMS/LENGTH_PENALTY match winner", winner)


if __name__ == "__main__":
    p = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "results" / "decode_search" / "ranking.json"
    )
    main(p)
