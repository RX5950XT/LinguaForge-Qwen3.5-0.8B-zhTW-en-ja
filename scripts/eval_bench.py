"""公開知識／常識基準 — 補上 eval_capability.py 自建 n=30 題組的外部可比性。

為什麼需要：`--axis general` 每語言只有 30 題，1 題就是 3.3 個百分點，
per-language 的 ±2 題全是雜訊；而且題目是自己出的，跟外界沒有共同刻度。

**汙染立場（重要）**：對 Qwen3.5 的預訓練語料而言，沒有任何公開基準能證明
未被汙染。但本專案量的是「同一份題目上 base → finetune 的差值」，汙染兩邊
共有、相減即抵消。所以絕對分數不得拿來宣稱模型能力，**只有 Δ 有意義**。

計分法：4 選 1，比對生成位置上 A/B/C/D 四個 token 的 logits 取最大者，
**同一題把選項輪轉 4 次各問一遍**（正解在每個位置各出現一次）。
不要求模型真的輸出合法格式 —— 這是刻意的：SFT 會傷害指令跟隨，若用自由生成
＋解析答案，會把「知識掉了」跟「格式跑掉」混成同一個數字。隨機基準 = 25%。

輪轉是必要的，不是講究：實測單輪時 base 押同一個字母 43%、v5f 63%（隨機應為
~27%），acc 直接被位置先驗污染，v5f 看起來掉 5 分但分不出是掉知識還是偏好變強。
另一條路（比選項文字的 logprob、題面完全不提字母）實測 base 只有 30/27.5/37.5
貼著隨機基準，選項是完整句子時 logprob 被表面形式主導，量不到正確性，已捨棄。

兩個軸：
  belebele  BELEBELE 900 題 × zho_Hant / jpn_Jpan / eng_Latn（CC-BY-SA-4.0）
            同一批題目的三語平行版本 → **跨語言可比**，閱讀理解／通用理解
  knowledge TMMLU+ (zh-TW 原生台灣考題) / MMLU (en) / MMMLU JA_JP (ja)，皆 MIT
            → 三者題目不同，**只有同語言內的 Δ 有效，不得跨語言比**

用法：
  uv run python scripts/eval_bench.py --tag base
  uv run python scripts/eval_bench.py --tag v5f --adapter outputs/sft-v5f
  uv run python scripts/eval_bench.py --tag v5f --adapter outputs/sft-v5f --bench belebele
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).parent.parent
MODEL_ID = "Qwen/Qwen3.5-0.8B"
LETTERS = ("A", "B", "C", "D")

BELEBELE_LANGS = {"zhtw": "zho_Hant", "ja": "jpn_Jpan", "en": "eng_Latn"}

# 作答指示用該語言寫，避免英文指示本身變成 en 的優勢
ASK = {
    "zhtw": "請從 A、B、C、D 中選出正確答案。只回答一個字母。\n答案：",
    "ja": "A、B、C、D の中から正しい答えを選んでください。アルファベット一文字だけで答えてください。\n答え：",
    "en": "Answer with a single letter from A, B, C, or D.\nAnswer:",
}


# ------------------------------------------------------------------ 題目建構

def mc_prompt(lang, question, options, passage=None):
    """letter 計分用的完整題面（含 A~D 選項列表與作答指示）。"""
    body = f"{passage.strip()}\n\n" if passage else ""
    opts = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, options))
    return f"{body}{question.strip()}\n{opts}\n\n{ASK[lang]}"


def item(lang, question, options, gold, passage=None):
    """letter_fn：輪轉時要用新的選項順序重建題面，所以題面必須是可重算的。"""
    opts = [str(o).strip() for o in options]

    def letter_fn(o):
        return mc_prompt(lang, question, o, passage)

    return {"letter": letter_fn(opts), "letter_fn": letter_fn,
            "options": opts, "gold": gold}


def from_belebele(lang, limit, seed):
    from datasets import load_dataset

    ds = load_dataset("facebook/belebele", BELEBELE_LANGS[lang], split="test")
    rows = list(ds)
    rows = subsample(rows, limit, seed)
    out = []
    for r in rows:
        opts = [r[f"mc_answer{i}"] for i in range(1, 5)]
        # correct_answer_num 是 "1".."4" 字串 → 0-based 索引
        gold = int(r["correct_answer_num"]) - 1
        out.append(item(lang, r["question"], opts, gold, r["flores_passage"]))
    return out


def from_knowledge(lang, limit, seed):
    """TMMLU+ / MMLU / MMMLU 欄位長得不一樣，各自轉成同一份 item。"""
    from datasets import get_dataset_config_names, load_dataset

    if lang == "zhtw":
        # TMMLU+ 沒有 all config，66 科各自是一個 config → 全載再抽樣
        subjects = get_dataset_config_names("ikala/tmmluplus")
        rows = []
        for s in subjects:
            for r in load_dataset("ikala/tmmluplus", s, split="test"):
                rows.append((r["question"], [r[k] for k in LETTERS], r["answer"]))
    elif lang == "en":
        ds = load_dataset("cais/mmlu", "all", split="test")
        rows = [(r["question"], r["choices"], LETTERS[r["answer"]]) for r in ds]
    else:
        ds = load_dataset("openai/MMMLU", "JA_JP", split="test")
        rows = [(r["Question"], [r[k] for k in LETTERS], r["Answer"]) for r in ds]

    rows = subsample(rows, limit, seed)
    return [item(lang, q, opts, LETTERS.index(ans.strip().upper()))
            for q, opts, ans in rows]


def subsample(rows, limit, seed):
    """固定 seed 抽樣：不同版本必須看同一批題目，否則 Δ 沒有意義。"""
    if limit and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return rows


# ------------------------------------------------------------------ 計分

def letter_ids(tok):
    """A/B/C/D 各自的單一 token id。非單 token 就直接炸掉，不要靜默用第一個。"""
    ids = []
    for L in LETTERS:
        enc = tok.encode(L, add_special_tokens=False)
        assert len(enc) == 1, f"{L!r} 不是單一 token: {enc}"
        ids.append(enc[0])
    return ids


@torch.no_grad()
def score_letter(tok, model, items, batch):
    """比對生成位置上 A/B/C/D 四個 token 的 logits。單次 forward，便宜。

    已知缺陷：0.8B 這種尺寸對「選項位置」有很強的先驗（實測 base 押同一個字母
    43%、v5f 63%，隨機應為 ~27%），acc 會被這個偏好污染。留著只當診斷用，
    正式數字看 score_content。
    """
    ids = torch.tensor(letter_ids(tok), device="cuda")
    picks = []
    for i in range(0, len(items), batch):
        convs = [[{"role": "user", "content": x["letter"]}]
                 for x in items[i:i + batch]]
        inp = tok.apply_chat_template(convs, add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt",
                                      padding=True).to("cuda")
        # padding_side="left" → 最後一個位置就是生成位置，對整個 batch 都成立。
        # logits_to_keep=1 不可省：vocab 248K，不設會實體化 [B, L, 248K] 整張
        # logits（B=16、L=800 就是 6.3GB），但這裡只用得到最後一格。
        logits = model(**inp, logits_to_keep=1).logits[:, -1, :]
        picks.extend(logits.index_select(1, ids).argmax(-1).tolist())
        print(f"    {min(i + batch, len(items))}/{len(items)}", flush=True)
    return picks


def rotate(x, r):
    """選項左旋 r 格；正解跟著搬到 (gold-r)%4。"""
    opts = [x["options"][(j + r) % 4] for j in range(4)]
    return {"letter": x["letter_fn"](opts), "gold": (x["gold"] - r) % 4}


def score_rotated(tok, model, items, batch):
    """同一題把選項輪轉 4 次各問一遍 → 正解在 A/B/C/D 各出現一次。

    位置先驗因此被完全攤平：模型就算固定押 A，四輪也只會答對其中一輪，
    acc 自動回到 25% 隨機基準，不會像單輪那樣被灌水或吃虧。
    每題 4 次 forward，但每次只讀一個 token 的 logits，仍比自由生成便宜。

    捨棄的做法：比選項文字的 logprob（cloze）——實測 base 只有 30/27.5/37.5，
    貼著 25% 隨機基準，選項是完整句子時 logprob 被表面形式主導、量不到正確性。
    """
    variants = [rotate(x, r) for r in range(4) for x in items]
    picks = score_letter(tok, model, variants, batch)
    golds = [v["gold"] for v in variants]
    return picks, golds


def run(tok, model, name, loader, limit, seed, batch, scoring):
    res = {}
    for lang in ("zhtw", "ja", "en"):
        items = loader(lang, limit, seed)
        print(f"  [{name}] {lang} n={len(items)}")
        if scoring == "rotate":
            picks, golds = score_rotated(tok, model, items, batch)
        else:
            picks = score_letter(tok, model, items, batch)
            golds = [x["gold"] for x in items]
        correct = sum(p == g for p, g in zip(picks, golds))
        acc = round(100 * correct / len(golds), 2)
        # 押同一個字母的比例：rotate 下這只反映先驗強度，不再影響 acc，
        # 但仍要存——它是判讀「模型有沒有在作答」的獨立訊號
        bias = max(picks.count(k) for k in range(4)) / len(picks)
        res[lang] = {"n": len(items), "correct": correct, "acc": acc,
                     "max_letter_share": round(bias, 3)}
        print(f"    acc={acc}%  最常選的字母佔比={bias:.1%}")
    res["avg"] = round(sum(res[l]["acc"] for l in ("zhtw", "ja", "en")) / 3, 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--bench", default="all",
                    choices=["all", "belebele", "knowledge"])
    ap.add_argument("--limit", type=int, default=900,
                    help="每語言題數上限（belebele 本來就只有 900）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--scoring", default="rotate", choices=["rotate", "letter"],
                    help="rotate=選項輪轉 4 次去除位置偏好（預設，正式數字用這個）；"
                         "letter=單輪，便宜但會被位置先驗污染，只當診斷")
    args = ap.parse_args()

    print("== loading model ==")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  adapter: {args.adapter}")
    model.eval()

    # 兩種計分法的分數不可互相比較 → 各自存一個檔，不共用 tag。
    # rotate 是正式數字，用無後綴檔名；letter 只是診斷，加後綴區隔。
    suffix = "" if args.scoring == "rotate" else f"-{args.scoring}"
    dest = ROOT / "results" / "bench" / f"{args.tag}{suffix}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    out |= {"tag": args.tag, "model": args.model, "adapter": args.adapter,
            "limit": args.limit, "seed": args.seed, "random_baseline": 25.0,
            "scoring": args.scoring}

    benches = ["belebele", "knowledge"] if args.bench == "all" else [args.bench]
    for b in benches:
        print(f"== {b} ==")
        out[b] = run(tok, model, b,
                     from_belebele if b == "belebele" else from_knowledge,
                     args.limit, args.seed, args.batch, args.scoring)

    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    print(f"\npeak VRAM: {out['peak_vram_gb']} GB")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results -> {dest}")


if __name__ == "__main__":
    main()
