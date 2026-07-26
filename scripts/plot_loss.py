"""畫 v3 訓練的 Training / Eval loss 雙曲線 → docs/assets/loss_curve.png。

matplotlib 不在主環境（保持訓練環境乾淨），用 --with 臨時掛載：
  uv run --with matplotlib python scripts/plot_loss.py
  uv run --with matplotlib python scripts/plot_loss.py --state <trainer_state.json> --out <png>
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
DEFAULT_STATE = ROOT / "results" / "v3" / "trainer_state.json"  # 進版控，outputs/ 不在
DEFAULT_OUT = ROOT / "docs" / "assets" / "loss_curve.png"
V2_FLOOR = 2.1634  # v2 eval_loss 地板（同基準比較見 CONTEXT）


def series(log, key):
    return [(x["step"], x[key]) for x in log if key in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    log = json.loads(args.state.read_text(encoding="utf-8"))["log_history"]
    train, ev = series(log, "loss"), series(log, "eval_loss")
    assert train and ev, f"missing series: train={len(train)} eval={len(ev)}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(*zip(*train), color="#3b7dd8", lw=1.6, label=f"Training loss ({len(train)} pts)")
    ax.plot(*zip(*ev), color="#e0523b", lw=2.0, marker="o", ms=5,
            label=f"Eval loss ({len(ev)} pts)")
    ax.axhline(V2_FLOOR, color="#999", ls="--", lw=1.0, label=f"v2 eval floor {V2_FLOOR}")

    fs, es = train[-1][1], ev[-1][1]
    ax.annotate(f"{es:.3f}", (ev[-1][0], es), textcoords="offset points",
                xytext=(-6, 8), color="#e0523b", fontsize=9, ha="right")
    ax.set(xlabel="Training step", ylabel="Loss",
           title="LinguaForge Qwen3.5-0.8B v3 — SFT loss (1 epoch, r64/α128 + NEFTune)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"train final {fs:.4f} | eval final {es:.4f} (< v2 {V2_FLOOR}) -> {args.out}")


if __name__ == "__main__":
    main()
