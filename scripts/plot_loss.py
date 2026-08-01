"""畫某一版訓練的 Training / Eval loss 雙曲線。

matplotlib 不在主環境（保持訓練環境乾淨），用 --with 臨時掛載：
  uv run --with matplotlib python scripts/plot_loss.py                      # 預設出貨版 v5e
  uv run --with matplotlib python scripts/plot_loss.py --state <trainer_state.json> --out <png>

trainer_state 一律從 `results/<版本>/` 讀（outputs/ 不進版控，刪了就沒了）。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
DEFAULT_STATE = ROOT / "results" / "v5" / "v5e-trainer_state.json"
DEFAULT_OUT = ROOT / "docs" / "assets" / "loss_curve.png"


def series(log, key):
    return [(x["step"], x[key]) for x in log if key in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--title", default="LinguaForge Qwen3.5-0.8B v5e — SFT loss "
                                       "(1 epoch, r64/α128, 502,993 samples)")
    # 地板線是選配：跨版本比 eval_loss 只有在 dev 集相同時才成立（見 D0）。
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--floor-label", default="previous best")
    args = ap.parse_args()

    log = json.loads(args.state.read_text(encoding="utf-8"))["log_history"]
    train, ev = series(log, "loss"), series(log, "eval_loss")
    assert train and ev, f"missing series: train={len(train)} eval={len(ev)}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(*zip(*train), color="#3b7dd8", lw=1.6, label=f"Training loss ({len(train)} pts)")
    ax.plot(*zip(*ev), color="#e0523b", lw=2.0, marker="o", ms=5,
            label=f"Eval loss ({len(ev)} pts)")
    if args.floor is not None:
        ax.axhline(args.floor, color="#999", ls="--", lw=1.0,
                   label=f"{args.floor_label} {args.floor}")

    fs, es = train[-1][1], ev[-1][1]
    ax.annotate(f"{es:.4f}", (ev[-1][0], es), textcoords="offset points",
                xytext=(-6, 8), color="#e0523b", fontsize=9, ha="right")
    ax.set(xlabel="Training step", ylabel="Loss", title=args.title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"train final {fs:.4f} | eval final {es:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
