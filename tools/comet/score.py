"""COMET 評分：讀 results/hyp/<tag>/*.txt，把分數寫回 results/<tag>.json。

用法（於專案根目錄）：
  uv run --project tools/comet python tools/comet/score.py --tag baseline
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MODEL = "Unbabel/wmt22-comet-da"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from comet import download_model, load_from_checkpoint

    ckpt = download_model(MODEL)
    model = load_from_checkpoint(ckpt)

    hyp_dir = ROOT / "results" / "hyp" / args.tag
    # 遞迴定位：results/ 已依版本分類；找得到就原地讀寫，找不到（新 tag）才回退根目錄
    result_file = next((ROOT / "results").rglob(f"{args.tag}.json"),
                       ROOT / "results" / f"{args.tag}.json")
    report = json.loads(result_file.read_text(encoding="utf-8"))

    for hyp_file in sorted(hyp_dir.glob("*.hyp.txt")):
        stem = hyp_file.name.removesuffix(".hyp.txt")
        read = lambda sfx: (hyp_dir / f"{stem}.{sfx}.txt").read_text(
            encoding="utf-8").rstrip("\n").split("\n")
        src, ref, hyp = read("src"), read("ref"), read("hyp")
        assert len(src) == len(ref) == len(hyp), stem
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
        out = model.predict(data, batch_size=args.batch, gpus=1)
        direction = stem.replace("2", "->", 1)
        report["results"][direction]["comet"] = round(out.system_score * 100, 2)
        print(f"{direction}: COMET {out.system_score * 100:.2f}")

    result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"updated -> {result_file}")


if __name__ == "__main__":
    main()
