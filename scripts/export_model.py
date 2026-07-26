"""Phase 5: 合併 LoRA → safetensors，並驗證合併後模型可正常翻譯。

GGUF 轉換另跑 llama.cpp 的 convert_hf_to_gguf.py（見 README）。

用法：
  uv run python scripts/export_model.py --adapter outputs/sft --out outputs/merged
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

# Windows console 預設 cp950 印不出日文長音符等字元；改 UTF-8 避免 verify 誤崩
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
CHECKS = [
    ("翻譯成繁體中文：", "The night market is crowded on weekends."),
    ("翻譯成日文：", "這家餐廳的牛肉麵非常好吃。"),
    ("Translate to English:", "台北捷運的班次很密集。"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=MODEL_ID)
    args = ap.parse_args()

    from peft import PeftModel

    print("== merging ==")
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    # 讓 merged 模型自帶正確 eos，下游（llama.cpp/GGUF/vLLM）不必手動傳
    eos_ids = sorted({tok.eos_token_id,
                      tok.convert_tokens_to_ids("<|im_end|>"),
                      tok.convert_tokens_to_ids("<|endoftext|>")} - {None, tok.unk_token_id})
    model.generation_config.eos_token_id = eos_ids
    model.generation_config.pad_token_id = tok.pad_token_id

    out = Path(args.out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"  saved -> {out}")

    print("== verify ==")
    model = model.cuda().eval()
    for instr, text in CHECKS:
        msgs = [{"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": f"{instr}\n{text}"}]
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                 eos_token_id=eos_ids, pad_token_id=tok.pad_token_id)
        res = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        assert res, f"empty output for: {text}"
        print(f"  [{instr}] {text}\n    -> {res}")
    print("EXPORT OK")


if __name__ == "__main__":
    main()
