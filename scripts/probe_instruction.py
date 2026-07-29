"""en→zhtw 對指令措辭有多敏感？

訓練資料把 zhtw 目標的 39,600 筆拆成 4 種指令變體（en/ja 只拆 3 種），
所以 evaluate.py 固定用的「翻譯成繁體中文：」只有 9,921 筆樣本，比 en 的對應
指令少 26%。若換個變體能改善漏譯，那 en→zhtw 輸 base 就有一部分是提示詞造成的。

用法：uv run python scripts/probe_instruction.py [--n 200] [--adapter outputs/sft-v5c]
"""
import argparse
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import (DECODE, SYSTEM, _SIMPLIFIED, batched_translate,  # noqa: E402
                      load_flores, score)

VARIANTS = ["翻譯成繁體中文：", "翻譯成臺灣正體中文：",
            "Translate to Traditional Chinese (Taiwan):",
            "台湾の繁体字中国語に翻訳してください："]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--adapter", default="outputs/sft-v5c")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from evaluate import MODEL_ID

    pairs = load_flores(args.n)
    src, ref = pairs[("en", "zhtw")]

    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    gen = {**DECODE["zhtw"], "num_beams": args.beams}
    print(f"n={len(src)}  decode={gen}  adapter={args.adapter}\n")
    print(f'{"指令":44}{"chrF++":>8}{"洩漏%":>8}{"長度比中位":>12}{"<0.75":>8}')
    for instr in VARIANTS:
        prompts = [f"{instr}\n{s}" for s in src]
        hyps = batched_translate(tok, model, prompts, args.batch, gen)
        m = score(("en", "zhtw"), hyps, ref)
        lr = [len(h) / max(1, len(r)) for h, r in zip(hyps, ref)]
        print(f"{instr:44}{m['chrf++']:8.2f}{m['simplified_leak_pct']:8.2f}"
              f"{st.median(lr):12.3f}{sum(x < 0.75 for x in lr):8}")
        out = Path(__file__).parent.parent / "results" / "hyp" / f"probe-{VARIANTS.index(instr)}"
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("src", src), ("ref", ref), ("hyp", hyps)):
            (out / f"en2zhtw.{name}.txt").write_text(
                "\n".join(r.replace("\n", " ") for r in rows) + "\n",
                encoding="utf-8", newline="\n")
    assert _SIMPLIFIED, "簡體字表不該是空的"


if __name__ == "__main__":
    main()
