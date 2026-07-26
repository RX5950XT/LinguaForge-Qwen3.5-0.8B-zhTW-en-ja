"""統計 SFT 資料的 token 量，用於估算訓練時間。"""

import json
from pathlib import Path

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
path = Path(__file__).parent.parent / "data" / "sft" / "train.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]

sample = rows[:5000]
texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in sample]
lens = [len(tok(t)["input_ids"]) for t in texts]
avg = sum(lens) / len(lens)
total = avg * len(rows)

print(f"rows: {len(rows):,}")
print(f"avg tokens/sample: {avg:.1f}  (p50 {sorted(lens)[len(lens)//2]}, "
      f"p95 {sorted(lens)[int(len(lens)*0.95)]}, max {max(lens)})")
print(f"est total tokens/epoch: {total/1e6:.1f}M")
for tps in (1405,):
    print(f"@ {tps} tok/s -> {total/tps/3600:.1f} h/epoch")
