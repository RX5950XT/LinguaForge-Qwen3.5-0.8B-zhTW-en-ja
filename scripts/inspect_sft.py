"""SFT JSONL 抽樣目檢 + 全量簡體殘留掃描。"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_data import has_simplified

path = Path(__file__).parent.parent / "data" / "sft" / "train.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
print(f"total: {len(rows):,}")

zh_targets = [r for r in rows
              if any(k in r["messages"][1]["content"][:30]
                     for k in ("繁體", "Traditional", "繁体字"))]
leaks = [r for r in zh_targets if has_simplified(r["messages"][2]["content"])]
print(f"zh-target rows: {len(zh_targets):,}, simplified residue: {len(leaks)}")

random.seed(7)
for r in random.sample(rows, 6):
    u = r["messages"][1]["content"]
    a = r["messages"][2]["content"]
    print(f"\nUSER: {u[:120]}")
    print(f"ASST: {a[:120]}")
