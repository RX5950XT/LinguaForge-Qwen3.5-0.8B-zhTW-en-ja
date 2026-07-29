"""在同一個句子子集上比較多個 tag 的 COMET。

tools/comet/score.py 只吃整份 result json，n 不同的兩次評測（--full 1012 vs
--limit 500）沒辦法直接比。這支一律取前 N 句，讓解碼設定對照是乾淨的。

用法（要在 COMET 隔離環境跑）：
  uv run --project tools/comet python scripts/comet_subset.py v5c-flores v5c-b4-flores
  uv run --project tools/comet python scripts/comet_subset.py --n 200 base-b4-flores v5c-b4-flores

第一個 tag 提供 src/ref（各 tag 的 src/ref 應相同，只有 hyp 不同）。
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIRECTIONS = ["en2zhtw", "zhtw2en", "en2ja", "ja2en", "ja2zhtw", "zhtw2ja"]


def read(tag: str, stem: str, kind: str, n: int) -> list[str] | None:
    p = ROOT / "results" / "hyp" / tag / f"{stem}.{kind}.txt"
    if not p.exists():
        return None
    lines = p.read_text(encoding="utf-8").rstrip("\n").split("\n")
    return lines[:n] if len(lines) >= n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="results/hyp/<tag>/ 底下的標籤")
    ap.add_argument("--n", type=int, default=500, help="取前 N 句（預設 500）")
    args = ap.parse_args()

    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    scores = {}
    for d in DIRECTIONS:
        src, ref = read(args.tags[0], d, "src", args.n), read(args.tags[0], d, "ref", args.n)
        if src is None or ref is None:
            print(f"  !! {args.tags[0]}/{d} 沒有 {args.n} 句 src/ref，跳過此方向")
            continue
        for tag in args.tags:
            hyp = read(tag, d, "hyp", args.n)
            if hyp is None:
                continue
            out = model.predict([{"src": a, "mt": c, "ref": b}
                                 for a, b, c in zip(src, ref, hyp)],
                                batch_size=32, gpus=1, progress_bar=False)
            scores[(tag, d)] = sum(out.scores) / len(out.scores) * 100

    w = max(16, max(len(t) for t in args.tags) + 2)
    print(f'\n{"dir":10}' + "".join(f"{t:>{w}}" for t in args.tags))
    for d in DIRECTIONS:
        cells = [scores.get((t, d)) for t in args.tags]
        print(f"{d:10}" + "".join(f"{v:{w}.2f}" if v is not None else f'{"-":>{w}}'
                                 for v in cells))
    print(f'{"AVG":10}' + "".join(
        f"{sum(v) / len(v):{w}.2f}" if (v := [scores[(t, d)] for d in DIRECTIONS
                                              if (t, d) in scores]) else f'{"-":>{w}}'
        for t in args.tags))


if __name__ == "__main__":
    main()
