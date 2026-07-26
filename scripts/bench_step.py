"""診斷：量測 LoRA 訓練單步 forward+backward 時間，區分 JIT 編譯與穩態速度。

用法：uv run python scripts/bench_step.py --bs 8 --seq 1024
"""

import argparse
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText

MODEL_ID = "Qwen/Qwen3.5-0.8B"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
           "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
           "out_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--nf4", action="store_true", help="QLoRA：NF4 4-bit 量化載入")
    args = ap.parse_args()

    print(f"config: model={args.model} bs={args.bs} seq={args.seq} "
          f"r={args.r} nf4={args.nf4} checkpointing={args.checkpointing}")
    kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.nf4:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    if not args.nf4:
        model = model.cuda()
    model.config.use_cache = False
    if args.nf4:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.checkpointing)
    elif args.checkpointing:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=TARGETS))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    vocab = model.config.get_text_config().vocab_size
    ids = torch.randint(0, vocab, (args.bs, args.seq), device="cuda")

    for i in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tps = args.bs * args.seq / dt
        print(f"  step {i}: {dt:7.2f}s  {tps:9.0f} tok/s  "
              f"VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
