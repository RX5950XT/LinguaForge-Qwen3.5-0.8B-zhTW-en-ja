"""把合併後的 bf16 全模型轉 GGUF 並量化（Q8_0 / Q4_K_M），可選匯出 MTP draft。

前置：先 `scripts/export_model.py` 合併出 release/merged-bf16，再跑本腳本。
llama.cpp 需自行 clone（本專案未內含）：
  git clone https://github.com/ggml-org/llama.cpp
  # 量化二進位可用官方預編 win 版（release 頁的 llama-*-bin-win-cpu-x64.zip）

用法：
  uv run python scripts/export_gguf.py \
    --llama-cpp <llama.cpp 目錄> --quantize-bin <llama-quantize.exe 路徑>
  # 加 --mtp 另匯出 MTP speculative draft

⚠️ 關鍵坑：Qwen3.5 有一顆 MTP 層（config mtp_num_hidden_layers=1）。預設轉換會把它併進主檔，
   導致 llama.cpp runtime 載入時報 `missing tensor 'blk.24.attn_norm.weight'`。
   → 主檔一律加 `--no-mtp` 轉出乾淨 trunk；MTP head 另用 `--mtp` 單獨匯出當 draft。
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
NAME = "linguaforge-v3-0.8b"


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, default=ROOT / "release" / "merged-bf16")
    ap.add_argument("--out", type=Path, default=ROOT / "release" / "gguf")
    ap.add_argument("--llama-cpp", type=Path, required=True, help="llama.cpp 目錄（含 convert_hf_to_gguf.py）")
    ap.add_argument("--quantize-bin", type=Path, required=True, help="llama-quantize(.exe) 路徑")
    ap.add_argument("--quants", nargs="+", default=["Q8_0", "Q4_K_M"])
    ap.add_argument("--mtp", action="store_true", help="另匯出 MTP head 當 speculative draft")
    args = ap.parse_args()

    convert = args.llama_cpp / "convert_hf_to_gguf.py"
    assert convert.exists(), f"找不到 {convert}"
    assert args.merged.exists(), f"找不到合併模型 {args.merged}（先跑 export_model.py）"
    args.out.mkdir(parents=True, exist_ok=True)
    f16 = args.out / f"{NAME}-f16.gguf"

    # 主檔：--no-mtp 轉乾淨 trunk（避開 blk.24 載入錯）
    run([sys.executable, str(convert), str(args.merged),
         "--outtype", "f16", "--no-mtp", "--outfile", str(f16)])

    for q in args.quants:
        run([str(args.quantize_bin), str(f16), str(args.out / f"{NAME}-{q}.gguf"), q])

    if args.mtp:  # MTP head 單獨匯出（含共享 embedding，體積偏大）
        run([sys.executable, str(convert), str(args.merged),
             "--mtp", "--outtype", "f16", "--outfile", str(args.out / f"{NAME}-mtp-f16.gguf")])

    print("GGUF EXPORT OK ->", args.out)


if __name__ == "__main__":
    main()
