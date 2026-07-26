"""多基準診斷計分板：讀 results/<tag>-<benchmark>.json → 方向 × 基準 矩陣。

紀律：**per-benchmark 分開看，絕不跨基準平均 COMET**（FLORES/NTREX 原生繁體 vs
WMT/ALT/TICO 的 s2twp 轉換參考不可比）。zhtw 目標在簡體基準的 reference-based 分數標 `~`（次要）。
給兩個 tag 時輸出 Δ（後者 − 前者）。

用法：
  uv run python scripts/scoreboard.py --tags v2                 # 單版多領域分佈（抓 FLORES-only 破綻）
  uv run python scripts/scoreboard.py --tags v2 v3 --metric comet
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DIRECTIONS = ["en->zhtw", "zhtw->en", "en->ja", "ja->en", "ja->zhtw", "zhtw->ja"]
BENCH_ORDER = ["flores", "ntrex", "wmt22", "alt", "tico19"]
NATIVE_ZH = {"flores", "ntrex"}          # 原生繁體參考
SIMPLIFIED_ZH = {"wmt22", "alt", "tico19"}  # s2twp 轉換參考 → zhtw 目標格次要


def load(tag):
    """{benchmark: {direction: {metric: val}}}"""
    out = {}
    for f in sorted(RESULTS.rglob(f"{tag}-*.json")):  # rglob：results/ 分類子目錄也讀得到
        bench = f.stem[len(tag) + 1:]
        out[bench] = json.loads(f.read_text(encoding="utf-8")).get("results", {})
    return out


def cell(data, bench, direction, metric):
    v = data.get(bench, {}).get(direction, {}).get(metric)
    return v


def is_secondary(bench, direction):
    """簡體基準的 zhtw 目標格 → reference-based 分數次要。"""
    return bench in SIMPLIFIED_ZH and direction.endswith("->zhtw")


def fmt(v, secondary):
    if v is None:
        return "   -  "
    s = f"{v:5.1f}"
    return f"{s}~" if secondary else f"{s} "


def matrix(title, tags, datasets, metric, benches):
    lines = [f"\n### {title}", ""]
    head = "方向".ljust(10) + "".join(b.ljust(9) for b in benches)
    lines.append(head)
    lines.append("-" * len(head))
    for d in DIRECTIONS:
        row = d.ljust(10)
        for b in benches:
            sec = is_secondary(b, d)
            if len(tags) == 1:
                row += fmt(cell(datasets[0], b, d, metric), sec).ljust(9)
            else:
                a = cell(datasets[0], b, d, metric)
                c = cell(datasets[-1], b, d, metric)
                if a is None or c is None:
                    row += "   -  ".ljust(9)
                else:
                    delta = c - a
                    row += (f"{delta:+5.1f}" + ("~" if sec else " ")).ljust(9)
        lines.append(row)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="1 個=分佈；2 個=Δ（後−前）")
    ap.add_argument("--metric", default="comet",
                    choices=["comet", "cometkiwi", "chrf++", "bleu"])
    args = ap.parse_args()

    datasets = [load(t) for t in args.tags]
    benches = [b for b in BENCH_ORDER if any(b in d for d in datasets)]
    if not benches:
        print("!! 找不到任何 results/<tag>-<benchmark>.json")
        return

    hdr = f"# Scoreboard — tags={args.tags}  metric={args.metric}"
    note = ("`~` = zhtw 目標在簡體基準（s2twp 轉換參考）→ reference-based 次要，"
            "主看 cometkiwi + 洩漏率。原生繁體參考：FLORES / NTREX。")
    out = [hdr, "", note]
    label = "Δ " + args.metric if len(args.tags) > 1 else args.metric
    out.append(matrix(f"{label}（方向 × 基準）", args.tags, datasets, args.metric, benches))
    out.append(matrix("簡體洩漏率 %（hyp-only，全領域有效）"
                      if len(args.tags) == 1 else "Δ 簡體洩漏率 %",
                      args.tags, datasets, "simplified_leak_pct", benches))
    text = "\n".join(out)
    print(text)
    (RESULTS / "scoreboard.md").write_text(text + "\n", encoding="utf-8")
    print(f"\n-> {RESULTS / 'scoreboard.md'}")


if __name__ == "__main__":
    main()
