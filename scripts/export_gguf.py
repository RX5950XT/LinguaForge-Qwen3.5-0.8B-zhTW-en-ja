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
NAME = "linguaforge-v5e-0.8b"   # 版本寫死會把舊檔覆蓋掉，用 --name 覆寫


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
    ap.add_argument("--name", default=NAME, help="輸出檔名前綴（含版本）")
    ap.add_argument("--mtp", action="store_true", help="另匯出 MTP head 當 speculative draft")
    args = ap.parse_args()

    convert = args.llama_cpp / "convert_hf_to_gguf.py"
    assert convert.exists(), f"找不到 {convert}"
    assert args.merged.exists(), f"找不到合併模型 {args.merged}（先跑 export_model.py）"
    args.out.mkdir(parents=True, exist_ok=True)
    f16 = args.out / f"{args.name}-f16.gguf"

    # 主檔要是乾淨 trunk：MTP 併進主檔的話 runtime 會報缺 blk.24.attn_norm.weight。
    # b10107 起用 --no-mtp 控制；更早的版本（實測 b8189）在 modify_tensors 裡無條件
    # `if name.startswith("mtp"): return`，本來就不會併進來，帶旗標反而 argparse 直接掛。
    help_txt = subprocess.run([sys.executable, str(convert), "--help"],
                              capture_output=True, text=True).stdout
    no_mtp = ["--no-mtp"] if "--no-mtp" in help_txt else []
    if not no_mtp:
        print("  note: 這版 convert_hf_to_gguf.py 沒有 --no-mtp（預設就跳過 MTP），不帶旗標")
    run([sys.executable, str(convert), str(args.merged),
         "--outtype", "f16", *no_mtp, "--outfile", str(f16)])

    for q in args.quants:
        run([str(args.quantize_bin), str(f16), str(args.out / f"{args.name}-{q}.gguf"), q])

    if args.mtp:  # MTP head 單獨匯出（含共享 embedding，體積偏大）
        assert "--mtp" in help_txt, "這版 convert_hf_to_gguf.py 不支援 --mtp，請升級 llama.cpp"
        run([sys.executable, str(convert), str(args.merged),
             "--mtp", "--outtype", "f16", "--outfile", str(args.out / f"{args.name}-mtp-f16.gguf")])

    print("GGUF EXPORT OK ->", args.out)


if __name__ == "__main__":
    main()
