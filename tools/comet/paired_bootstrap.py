"""兩個系統在同一方向上的 COMET paired bootstrap，判 Δ 是不是雜訊。

system_score 只是 segment 分數的平均，兩版差 0.1 到底算不算贏，不看 CI 是猜的。
配對重抽（兩系統用同一組抽樣索引）才對得起「同一批句子」這個設定——
非配對的 CI 會把句子難度的變異也算進去，寬到什麼都測不出來。

用法（於專案根目錄）：
  uv run --project tools/comet python tools/comet/paired_bootstrap.py \
    --a base-full-flores --b v5e-flores --direction en2zhtw
"""

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MODEL = "Unbabel/wmt22-comet-da"


def read_dir(tag, stem):
    d = ROOT / "results" / "hyp" / tag
    out = []
    for sfx in ("src", "ref", "hyp"):
        p = d / f"{stem}.{sfx}.txt"
        assert p.exists(), f"找不到 {p}"
        out.append(p.read_text(encoding="utf-8").rstrip("\n").split("\n"))
    src, ref, hyp = out
    assert len(src) == len(ref) == len(hyp), f"{tag}/{stem} 三檔行數不一致"
    return src, ref, hyp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="系統 A 的 tag（基準）")
    ap.add_argument("--b", required=True, help="系統 B 的 tag（候選）")
    ap.add_argument("--direction", required=True, help="如 en2zhtw")
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    src_a, ref_a, hyp_a = read_dir(args.a, args.direction)
    src_b, ref_b, hyp_b = read_dir(args.b, args.direction)
    # 配對的前提：同一份 src/ref。不相等就不是同一批句子，Δ 沒有意義
    assert src_a == src_b, "兩個 tag 的 src 不同，無法配對比較"
    assert ref_a == ref_b, "兩個 tag 的 ref 不同，無法配對比較"

    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(MODEL))

    def seg_scores(hyp):
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src_a, hyp, ref_a)]
        return model.predict(data, batch_size=args.batch, gpus=1).scores

    sa, sb = seg_scores(hyp_a), seg_scores(hyp_b)
    n = len(sa)
    diff = sum(b - a for a, b in zip(sa, sb)) / n * 100

    rng = random.Random(args.seed)
    deltas, wins = [], 0
    for _ in range(args.resamples):
        idx = [rng.randrange(n) for _ in range(n)]          # 同一組索引套兩邊 = 配對
        d = sum(sb[i] - sa[i] for i in idx) / n * 100
        deltas.append(d)
        wins += d > 0
    deltas.sort()
    lo, hi = deltas[int(0.025 * args.resamples)], deltas[int(0.975 * args.resamples)]
    p_b_better = wins / args.resamples

    print(f"\n{args.direction}  n={n}  resamples={args.resamples}")
    print(f"  {args.a:<20} COMET {sum(sa)/n*100:.2f}")
    print(f"  {args.b:<20} COMET {sum(sb)/n*100:.2f}")
    print(f"  delta(B-A) = {diff:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  P(B 較佳) = {p_b_better:.3f}")
    verdict = ("B 顯著較佳" if lo > 0 else
               "A 顯著較佳" if hi < 0 else
               "CI 跨 0 → 差異與雜訊不可區分")
    print(f"  結論：{verdict}")


if __name__ == "__main__":
    main()
