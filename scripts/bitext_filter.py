"""LaBSE 對位分數快取 → data/labse/<corpus>.npz（lineno, score）。

為什麼需要語意過濾：`prepare_data.py` 只做規則式清洗（長度比、語言驗證、標點對稱），
對「兩句根本沒關係」完全束手無策。實測見 docs/RESEARCH-v5.md F7——
en→zhtw 配方有 17.6% 的樣本 LaBSE 相似度 < 0.60，en→ja 只有 8.0%，
正好對上「en→zhtw 是唯一輸給 base 的方向」。`pre_globalvoices` 的
ponytail 註解本來就寫著「錯位殘留靠低 cap 壓制；仍崩再上 LaBSE/COMET-QE」。

用 `setu4993/LaBSE`（BertModel，`pooler_output` 即句向量）而非 sentence-transformers：
後者會把 transformers 往下釘，主環境必須留在 5.x 才載得動 Qwen3.5。

分數寫成快取而不是在 prepare_data 裡即時算，是為了讓資料準備維持純 CPU、
可重跑、可目檢；GPU 只在這裡用一次。

用法：
  uv run python scripts/bitext_filter.py                      # 全語料
  uv run python scripts/bitext_filter.py --only coct.en-zhtw.tsv --device cpu
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from prepare_data import CORPORA, ROOT, load_corpus, load_eval_lines

MODEL_ID = "setu4993/LaBSE"
OUT = ROOT / "data" / "labse"
MAX_LEN = 128       # 訓練樣本 p99 約 338 token，但對位訊號前 128 個就夠


@torch.no_grad()
def score_pairs(rows, tok, model, device, bs):
    """逐批算完立刻收斂成純量，GPU 記憶體是 O(batch) 而非 O(語料)。

    別改回「先把整份語料 embed 完再一次相乘」：opensub.en-zhtw 清洗後 103 萬列，
    768 維 float32 兩側同時在手就是 6 GB，驅動的 sysmem fallback 會默默溢到主記憶體，
    不報錯、只是慢十倍。
    """
    def emb(texts):
        b = tok(texts, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN).to(device)
        return torch.nn.functional.normalize(model(**b).pooler_output.float(), dim=-1)

    out = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        s = (emb([r[1] for r in chunk]) * emb([r[2] for r in chunk])).sum(-1)
        out.append(s.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="只處理指定檔名")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="重算已存在的快取")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL_ID, dtype=dtype).to(args.device).eval()

    stats, eval_lines = Counter(), load_eval_lines()
    for fname, l1, l2 in CORPORA:
        if args.only and fname not in args.only:
            continue
        dst = OUT / f"{fname}.npz"
        if dst.exists() and not args.force:
            print(f"  {fname}: 快取已存在，跳過")
            continue
        rows = load_corpus(fname, l1, l2, stats, eval_lines)
        if not rows:
            continue
        s = score_pairs(rows, tok, model, args.device, args.batch_size)
        np.savez(dst, lineno=np.array([r[0] for r in rows], dtype=np.int64), score=s)
        print(f"  -> {dst.name}  中位 {np.median(s):.3f}  "
              f"<0.60 {(s < 0.60).mean():.1%}  <0.70 {(s < 0.70).mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
