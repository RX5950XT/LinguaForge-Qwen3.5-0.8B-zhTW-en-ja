"""回歸護欄：候選版在 FLORES（原生繁體錨點）上不得跌破 v2 戰果。

檢查（對 baseline FLORES 動態比對，附容忍度吸收 bootstrap 雜訊）：
- 簡體洩漏（硬閘，錨點）：en→zhtw、ja→zhtw ≤ baseline + 0.3
- 語意 COMET（不回吐）：en→ja、ja→en、ja→zhtw、zhtw→ja ≥ baseline − 0.5
任一 FAIL → exit 1。

用法：
  uv run python scripts/regression_guard.py --candidate v3            # 對 v2-flores 比
  uv run python scripts/regression_guard.py --candidate v3 --baseline v2
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"

# (direction, metric, mode, tol)  mode: "max"=不得高於 baseline+tol；"min"=不得低於 baseline-tol
CHECKS = [
    ("en->zhtw", "simplified_leak_pct", "max", 0.3),
    ("ja->zhtw", "simplified_leak_pct", "max", 0.3),
    ("en->ja", "comet", "min", 0.5),
    ("ja->en", "comet", "min", 0.5),
    ("ja->zhtw", "comet", "min", 0.5),
    ("zhtw->ja", "comet", "min", 0.5),
]


def load(tag):
    f = next(RESULTS.rglob(f"{tag}-flores.json"), None)  # rglob：容許 results/ 分類子目錄
    if f is None:
        sys.exit(f"!! 缺 {tag}-flores.json（先跑 evaluate.py --tag {tag} --benchmark flores）")
    return json.loads(f.read_text(encoding="utf-8"))["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="v2")
    args = ap.parse_args()

    base, cand = load(args.baseline), load(args.candidate)
    print(f"# Regression guard: {args.candidate} vs {args.baseline} (FLORES)\n")
    print(f"{'direction':10} {'metric':20} {'base':>7} {'cand':>7} {'bound':>8}  verdict")
    print("-" * 66)

    failed = False
    for d, m, mode, tol in CHECKS:
        b = base.get(d, {}).get(m)
        c = cand.get(d, {}).get(m)
        if b is None or c is None:
            print(f"{d:10} {m:20} {'?':>7} {'?':>7} {'':>8}  SKIP(缺值)")
            continue
        if mode == "max":
            bound, ok = b + tol, c <= b + tol
        else:
            bound, ok = b - tol, c >= b - tol
        failed |= not ok
        print(f"{d:10} {m:20} {b:7.2f} {c:7.2f} {bound:8.2f}  {'PASS' if ok else 'FAIL <<<'}")

    print()
    if failed:
        sys.exit("RESULT: FAIL — 候選版跌破 v2 底線，不得出貨")
    print("RESULT: PASS — v2 底線全數守住")


if __name__ == "__main__":
    main()
