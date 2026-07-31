"""量 GGUF 各精度的翻譯品質，用 llama-cli 逐句跑 FLORES 子集。

為什麼要單獨一支：`evaluate.py` 走 transformers，量不到 GGUF 這條出貨路徑。
先前只用 export_model.py 裡寫死的三句 smoke test 判「量化沒問題」，
那三句只證明得了「沒有 thinking 雜訊、沒有簡體洩漏」，證明不了品質。

三種精度必須同 runtime、同解碼跑，才分得出「量化稅」與「解碼差異」。
注意：llama-cli 沒有 beam search，所以這裡的絕對分數**不能**跟 evaluate.py
的 beam=4 數字比，只能拿來比同一批設定下 f16 / Q8_0 / Q4_K_M 三者的相對差。

CJK 提示詞一律寫檔再用 -f 餵：Windows 主控台 cp950 會打壞 -p 的 UTF-8 內容。

用法：
  uv run python scripts/eval_gguf.py --direction en2zhtw --limit 100
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
BIN = Path("D:/Workspace/AI_inference/Qwen-35BA3B-RTX3070Ti/bin/llama-cli.exe")
GGUF_DIR = ROOT / "release" / "gguf-v5e"
HYP_DIR = ROOT / "results" / "hyp" / "v5e-flores"
SYSTEM = "You are a professional translator."
INSTR = {"zhtw": "翻譯成繁體中文：", "en": "Translate to English:", "ja": "翻譯成日文："}


def target_of(direction):
    return direction.split("2", 1)[1]


def run_one(model, prompt_file, max_new):
    out = subprocess.run(
        [str(BIN), "-m", str(model), "--jinja", "-st", "-ngl", "99",
         "-n", str(max_new), "--temp", "0",
         # 舊的 --chat-template-kwargs enable_thinking 已被靜默忽略，thinking 會洩進譯文
         "--reasoning", "off", "--reasoning-budget", "0",
         "-sys", SYSTEM, "-f", str(prompt_file)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = (out.stdout + out.stderr).split("\n")
    try:
        s = next(n for n, l in enumerate(lines) if l.startswith("> "))
        e = next(n for n, l in enumerate(lines) if l.startswith("[ Prompt:"))
    except StopIteration:
        return ""
    # "> " 之後是回顯的提示詞（可能多行），真正的輸出是空行之後那段
    body = [l for l in lines[s + 1:e] if l.strip()]
    return body[-1].strip() if body else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="en2zhtw")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--quants", nargs="+", default=["f16", "Q8_0", "Q4_K_M"])
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    import sacrebleu
    sys.path.insert(0, str(ROOT / "scripts"))
    from evaluate import _SIMPLIFIED                   # 洩漏字集只有一個來源，不重寫

    tgt = target_of(args.direction)
    read = lambda s: (HYP_DIR / f"{args.direction}.{s}.txt").read_text(
        encoding="utf-8").rstrip("\n").split("\n")
    src, ref = read("src")[:args.limit], read("ref")[:args.limit]

    tmp = ROOT / ".gguf_eval_prompt.txt"
    rows = {}
    for q in args.quants:
        model = GGUF_DIR / f"linguaforge-v5e-0.8b-{q}.gguf"
        assert model.exists(), f"找不到 {model}"
        hyps = []
        for i, s in enumerate(src, 1):
            tmp.write_text(f"{INSTR[tgt]}\n{s}", encoding="utf-8", newline="\n")
            hyps.append(run_one(model, tmp, args.max_new))
            if i % 20 == 0:
                print(f"  [{q}] {i}/{len(src)}", flush=True)
        rows[q] = hyps
        (GGUF_DIR / f"{args.direction}.{q}.hyp.txt").write_text(
            "\n".join(hyps), encoding="utf-8", newline="\n")
    tmp.unlink(missing_ok=True)

    # 把評測路徑（transformers bf16 + beam=4 + 逐語言 nrng）算在同一批句子上當對照。
    # llama-cli 沒有 beam search 也沒有 no_repeat_ngram，出貨路徑跑的不是被評測的解碼設定，
    # 這一列就是「出貨路徑 vs 評測路徑」的實際落差。
    rows["bf16-beam4*"] = read("hyp")[:args.limit]

    lang = {"zhtw": "zh", "ja": "ja-mecab", "en": "13a"}[tgt]
    print(f"\n{args.direction}  n={len(src)}")
    print("  * bf16-beam4 來自 results/hyp/（transformers，beam=4 + 逐語言 nrng），非本次跑")
    print(f"{'系統':<13} {'chrF++':>7} {'BLEU':>7} {'空輸出':>7} {'洩漏%':>7} {'與f16相同':>10}")
    base = rows.get("f16")
    for q, hyps in rows.items():
        chrf = sacrebleu.corpus_chrf(hyps, [ref], word_order=2).score
        bleu = sacrebleu.corpus_bleu(hyps, [ref], tokenize=lang).score
        empty = sum(not h.strip() for h in hyps)
        leak = (sum(any(c in _SIMPLIFIED for c in h) for h in hyps) / len(hyps) * 100
                if tgt == "zhtw" else float("nan"))
        same = (sum(a == b for a, b in zip(base, hyps)) / len(hyps) * 100
                if base else float("nan"))
        print(f"{q:<8} {chrf:7.2f} {bleu:7.2f} {empty:7d} {leak:7.2f} {same:9.1f}%")


if __name__ == "__main__":
    main()
