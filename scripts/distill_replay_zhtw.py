"""補繁中 replay 缺口：把英文 replay 樣本整組翻成 zh-TW，附加到 replay.jsonl。

為什麼要這樣做：公開語料裡沒有「授權乾淨 + 非機翻 + 足量」的繁中指令集
（見 docs/RESEARCH-v4.md §4.1），build_replay.py 只湊得到 1.3K 繁中，
相對 en/ja 的 16~17K 嚴重不足。

為什麼是翻譯而不是讓 teacher 自己作答：oasst2 的答案是人寫的，品質高於 2B 現場生成；
而「官方 Qwen3.5-2B + s2twp」正是本專案已驗證的最強翻譯組合（COMET 88.21、洩漏 0.2%）。
翻譯人寫答案 ⇒ 保住答案品質，只借用 teacher 的翻譯能力。

輸出一律 s2twp 後複檢，洩漏樣本直接丟——replay 不能反過來把簡體教回去。

用法：
  uv run python scripts/distill_replay_zhtw.py --n 200        # 先小量測吞吐
  uv run python scripts/distill_replay_zhtw.py --n 12000
"""

import argparse
import json
import time
from pathlib import Path

import torch
from opencc import OpenCC
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).parent.parent
REPLAY = ROOT / "data" / "sft" / "replay.jsonl"
OUT = ROOT / "data" / "sft" / "replay_zhtw_distilled.jsonl"

SYSTEM = "You are a professional translator."
INSTR = "翻譯成繁體中文："
cc_s2twp = OpenCC("s2twp")
cc_s2tw = OpenCC("s2tw")

CJK_RE = __import__("re").compile("[一-鿿]")


def stop_token_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(ids)


def translate(tok, model, texts, batch, max_new):
    eos = stop_token_ids(tok)
    out, t0, done = [], time.time(), 0
    for i in range(0, len(texts), batch):
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{INSTR}\n{t}"}]
                 for t in texts[i:i + batch]]
        inp = tok.apply_chat_template(convs, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt",
                                      padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id)
        n = inp["input_ids"].shape[1]
        chunk = [tok.decode(x[n:], skip_special_tokens=True).strip() for x in g]
        out.extend(chunk)
        done += sum(len(x) for x in chunk)
        el = time.time() - t0
        print(f"  {len(out)}/{len(texts)}  {done/max(el,1):.0f} char/s  "
              f"eta {(len(texts)-len(out))*el/max(len(out),1)/60:.0f} min", flush=True)
    return out


def usable(zh: str) -> bool:
    """譯文要有中文、無簡體殘留、長度合理。"""
    return bool(zh) and 5 <= len(zh) <= 4000 and CJK_RE.search(zh) is not None \
        and cc_s2tw.convert(zh) == zh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000, help="要翻幾組英文 replay")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--max-answer-chars", type=int, default=1200,
                    help="超長答案跳過，避免單筆吃掉整批時間")
    args = ap.parse_args()

    rows = [json.loads(x) for x in REPLAY.read_text(encoding="utf-8").rstrip("\n").split("\n") if x]
    # 只拿英文樣本（判據：兩側都沒有 CJK）
    en = [r for r in rows
          if not CJK_RE.search(r["messages"][0]["content"] + r["messages"][1]["content"])]
    en = [r for r in en
          if len(r["messages"][1]["content"]) <= args.max_answer_chars][:args.n]
    print(f"英文 replay 可用 {len(en):,} 組（上限 {args.n:,}）")

    tok = AutoTokenizer.from_pretrained(args.teacher, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.teacher, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    print("== 翻 prompt ==")
    zh_q = translate(tok, model, [r["messages"][0]["content"] for r in en],
                     args.batch, 256)
    print("== 翻 answer ==")
    zh_a = translate(tok, model, [r["messages"][1]["content"] for r in en],
                     args.batch, args.max_new)

    kept, dropped = [], 0
    for q, a in zip(zh_q, zh_a, strict=True):
        q, a = cc_s2twp.convert(q), cc_s2twp.convert(a)
        if usable(q) and usable(a):
            kept.append({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": a}]})
        else:
            dropped += 1
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT} ({len(kept):,} 保留 / {dropped:,} 丟棄)")
    print("附加進 replay：")
    print(f"  Get-Content {OUT.name} | Add-Content {REPLAY.name}   # 或直接 cat 合併")


if __name__ == "__main__":
    main()
