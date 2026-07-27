"""用修正後的偵測器（s2tw）重算 results/**.json 裡的 simplified_leak_pct。

原本用 OpenCC `s2t` 判洩漏，但 s2t 會把「剛才→剛纔」「人群→人羣」「稽核→稽覈」
這些正確的台灣用字轉成傳統異體字，於是被判成「有簡體」——偽陽性。
`s2tw` 以台灣標準為目標，不做這些異體轉換，才是正確判據。

譯文都還在 results/hyp/<tag>/<dir>.hyp.txt，直接重算即可，不必重跑模型。

用法：
  uv run python scripts/rescore_leak.py --dry-run   # 只列差異
  uv run python scripts/rescore_leak.py             # 寫回 JSON
"""

import argparse
import json
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
cc_s2tw = OpenCC("s2tw")


def leak_of(hyp_file: Path):
    hyps = [l for l in hyp_file.read_text(encoding="utf-8").rstrip("\n").split("\n") if l.strip()]
    if not hyps:
        return None
    return round(sum(cc_s2tw.convert(h) != h for h in hyps) / len(hyps) * 100, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed, missing = [], []
    for jf in sorted(RESULTS.rglob("*.json")):
        if jf.parent.name == "capability" or "hyp" in jf.parts:
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):   # 明細檔（list）不是評分 JSON
            continue
        results = data.get("results")
        if not isinstance(results, dict):
            continue
        tag = data.get("tag", jf.stem)
        dirty = False
        for name, metrics in results.items():
            if not isinstance(metrics, dict) or metrics.get("simplified_leak_pct") is None:
                continue
            hyp = RESULTS / "hyp" / tag / f"{name.replace('->', '2')}.hyp.txt"
            if not hyp.exists():
                missing.append(f"{tag}/{name}")
                continue
            new = leak_of(hyp)
            old = metrics["simplified_leak_pct"]
            if new is not None and abs(new - old) > 1e-9:
                changed.append((tag, name, old, new))
                metrics["simplified_leak_pct"] = new
                dirty = True
        if dirty and not args.dry_run:
            jf.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    print(f"{'tag':<16}{'direction':<12}{'舊(s2t)':>10}{'新(s2tw)':>10}{'偽陽性':>9}")
    for tag, name, old, new in changed:
        print(f"{tag:<16}{name:<12}{old:>9.2f}%{new:>9.2f}%{old - new:>8.2f}")
    print(f"\n{len(changed)} 筆更新" + ("（dry-run，未寫檔）" if args.dry_run else ""))
    if missing:
        print(f"!! 找不到 hyp 檔，未重算：{len(missing)} 筆 — {missing[:5]}")


if __name__ == "__main__":
    main()
