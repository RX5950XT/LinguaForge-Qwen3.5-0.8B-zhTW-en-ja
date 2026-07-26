"""B1 污染閘資料：把全部 eval 基準（flores/ntrex/wmt22/alt/tico19）的每一句
src/tgt「個別語言行」normalize 後寫入 data/eval_lines.txt。

prepare_data.py 讀此檔，訓練 pair 任一側命中即丟（防「背考卷」）。
每行同時收錄 s2twp 變體：訓練側簡體行會被 s2twp 轉換後才比對，
沒有變體會漏掉「同一句、不同字形」的碰撞。

用法：
  uv run python scripts/dump_eval_lines.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import BENCHMARKS  # noqa: E402  各 loader 自行下載/快取基準
from prepare_data import cc_s2twp, norm  # noqa: E402

OUT = Path(__file__).parent.parent / "data" / "eval_lines.txt"


def main():
    lines = set()
    for name, loader in BENCHMARKS.items():
        before = len(lines)
        for (src, tgt), (src_sents, refs) in loader(None).items():  # 全量，不截斷
            for s in (*src_sents, *refs):
                s = norm(s)
                if s:
                    lines.add(s)
                    lines.add(norm(cc_s2twp.convert(s)))
        print(f"  {name}: +{len(lines) - before:,}")
    OUT.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
    print(f"-> {OUT} ({len(lines):,} lines)")


if __name__ == "__main__":
    main()
