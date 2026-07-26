"""CometKiwi（reference-free QE）評分：讀 results/hyp/<tag>/*.txt 的 src+hyp，
把分數寫回 results/<tag>.json 的 "cometkiwi" 欄（不需參考譯文）。

用途：對 s2twp 轉換參考的簡體基準（WMT/ALT/TICO）的 zhtw 目標格，
reference-based COMET 受轉換誤差污染 → 改用 CometKiwi 當主要語意指標。

模型 Unbabel/wmt22-cometkiwi-da 為 gated + cc-by-nc-sa（非商用，僅供評測），
需先以 HF token 接受授權（huggingface-cli login / HF_TOKEN）。放隔離環境（transformers<4.58）。

用法（於專案根目錄）：
  uv run --project tools/comet python tools/comet/kiwi.py --tag v2-ntrex
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MODEL = "Unbabel/wmt22-cometkiwi-da"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from comet import download_model, load_from_checkpoint

    ckpt = download_model(MODEL)  # gated：未接受授權會在此報 401/403
    model = load_from_checkpoint(ckpt)

    hyp_dir = ROOT / "results" / "hyp" / args.tag
    # 遞迴定位：results/ 已依版本分類；找得到就原地讀寫，找不到（新 tag）才回退根目錄
    result_file = next((ROOT / "results").rglob(f"{args.tag}.json"),
                       ROOT / "results" / f"{args.tag}.json")
    report = json.loads(result_file.read_text(encoding="utf-8"))

    for hyp_file in sorted(hyp_dir.glob("*.hyp.txt")):
        stem = hyp_file.name.removesuffix(".hyp.txt")
        read = lambda sfx: (hyp_dir / f"{stem}.{sfx}.txt").read_text(
            encoding="utf-8").splitlines()
        src, hyp = read("src"), read("hyp")
        assert len(src) == len(hyp), stem
        data = [{"src": s, "mt": h} for s, h in zip(src, hyp)]  # 無 ref
        out = model.predict(data, batch_size=args.batch, gpus=1)
        direction = stem.replace("2", "->", 1)
        report["results"][direction]["cometkiwi"] = round(out.system_score * 100, 2)
        print(f"{direction}: CometKiwi {out.system_score * 100:.2f}")

    result_file.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"updated -> {result_file}")


if __name__ == "__main__":
    main()
