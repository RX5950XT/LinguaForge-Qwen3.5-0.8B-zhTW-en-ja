"""Phase 3: LoRA SFT 訓練（TRL SFTTrainer，prompt 遮罩、packing）。

用法：
  uv run python scripts/train_sft.py                      # 讀 configs/sft_lora.yaml
  uv run python scripts/train_sft.py --max-steps 200      # 子集試訓
"""

import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).parent.parent


def to_prompt_completion(row):
    msgs = row["messages"]
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "sft_lora.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="試訓步數上限")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    t = cfg["train"]
    out_dir = args.output_dir or cfg["output_dir"]

    print("== data ==")
    ds = load_dataset("json", data_files={
        "train": str(ROOT / cfg["train_file"]),
        "dev": str(ROOT / cfg["dev_file"]),
    })
    ds = ds.map(to_prompt_completion, remove_columns=["messages"])
    print(ds)

    print("== model ==")
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if cfg.get("quant") == "nf4":  # v3: 2B QLoRA 診斷分支
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(cfg["model_id"], **kwargs)
    model.config.use_cache = False

    lora = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        task_type="CAUSAL_LM", target_modules=lora["target_modules"],
    )

    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=t["num_train_epochs"],
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        max_length=t["max_length"],
        packing=t["packing"],
        gradient_checkpointing=t["gradient_checkpointing"],
        optim=t["optim"],
        neftune_noise_alpha=t.get("neftune_noise_alpha"),
        bf16=True,
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        seed=t["seed"],
        report_to=[],
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model, args=sft_cfg, processing_class=tok, peft_config=peft_cfg,
        train_dataset=ds["train"], eval_dataset=ds["dev"],
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
