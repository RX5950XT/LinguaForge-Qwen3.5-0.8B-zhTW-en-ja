"""Phase 3: LoRA SFT 訓練（TRL SFTTrainer，prompt 遮罩、packing）。

用法：
  uv run python scripts/train_sft.py                      # 讀 configs/sft_lora.yaml
  uv run python scripts/train_sft.py --max-steps 200      # 子集試訓
"""

import argparse
import random
from pathlib import Path

import pyarrow.compute as pc
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).parent.parent


def to_prompt_completion(row):
    msgs = row["messages"]
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


class TokenBudgetBatches:
    """依 token 預算組 micro-batch（長度相近者同批），每個 epoch 重洗批次順序。

    vocab 248K 讓 logits 吃掉 ~3.3MB/token，8GB 卡每個 micro-batch 只裝得下 ~1450 token
    （實測 bs1×1450 / bs4×362 / bs11×132 都是 6.90GB）。而單步時間幾乎與 token 數無關：
    256 token 只有 300 tok/s，1450 token 有 1300~1600 tok/s。
    固定 batch size 得遷就最長樣本（bs2），中位數 88 token 的短樣本就白燒 GPU；
    改成填滿 token 預算後整體快 4~5 倍。
    """

    def __init__(self, lengths, budget, seed):
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batches, cur, cur_max = [], [], 0
        for i in order:
            m = max(cur_max, lengths[i])
            if cur and m * (len(cur) + 1) > budget:
                self.batches.append(cur)
                cur, cur_max, m = [], 0, lengths[i]
            cur.append(i)
            cur_max = m
        if cur:
            self.batches.append(cur)
        self.rng = random.Random(seed)
        self.rng.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        self.rng.shuffle(self.batches)
        return iter(self.batches)


class TokenBudgetSFTTrainer(SFTTrainer):
    token_budget = 1450

    def get_train_dataloader(self):
        ds = self._remove_unused_columns(self.train_dataset, description="Training")
        lengths = pc.list_value_length(ds.data.column("input_ids")).to_pylist()
        bs = TokenBudgetBatches(lengths, self.token_budget, self.args.seed)
        print(f"  token-budget batches: {len(bs):,} micro-batch/epoch "
              f"(平均 {len(lengths) / len(bs):.1f} 筆/批)")
        return self.accelerator.prepare(DataLoader(
            ds, batch_sampler=bs, collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=ROOT / "configs" / "sft_lora.yaml")
    ap.add_argument("--max-steps", type=int, default=None, help="試訓步數上限")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    t = cfg["train"]
    out_dir = args.output_dir or cfg["output_dir"]

    tok = AutoTokenizer.from_pretrained(cfg["model_id"])

    print("== data ==")
    ds = load_dataset("json", data_files={
        "train": str(ROOT / cfg["train_file"]),
        "dev": str(ROOT / cfg["dev_file"]),
    })
    ds = ds.map(to_prompt_completion, remove_columns=["messages"])

    # 超過 max_length 的樣本整筆丟掉，不讓 TRL 截斷：被截斷的目標是硬切在半句，
    # 等於在示範「講到一半停」——正是 v3 長篇腰斬要修的毛病。
    def fits(row):
        text = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
        return len(tok(text)["input_ids"]) <= t["max_length"]

    before = {k: len(v) for k, v in ds.items()}
    ds = ds.filter(fits, num_proc=4)
    for k, n in before.items():
        print(f"  {k}: {len(ds[k]):,} / {n:,}  (超長丟棄 {n - len(ds[k]):,})")
    print(ds)

    print("== model ==")
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
        # packing 必須關：TRL 的 bfd packing 會自動開 padding-free，而那只支援 FlashAttention。
        # 用 sdpa 時同一條打包序列裡的樣本會互相 attend（實測：改前一個樣本會讓後一個樣本的
        # logits 變動 6.6），等於在教模型「講完一段可以接上不相干的內容」。
        # Qwen3.5 的 linear attention 層更糟——遞迴狀態直接跨越樣本邊界，裝了 flash-attn 也修不掉。
        packing=t["packing"],
        # 不打包的 padding 浪費由 TokenBudgetBatches 處理（同批長度相近），
        # 這裡的 sampler 不會被用到——get_train_dataloader 已整個換掉。
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
        # v3 是訓完才發現整體退化。這次讓 trainer 自己留 eval_loss 最低的權重，
        # 最後存檔就是最佳點而不是最後一步。
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=t["seed"],
        report_to=[],
        dataset_num_proc=1,
    )

    TokenBudgetSFTTrainer.token_budget = t["token_budget"]
    trainer = TokenBudgetSFTTrainer(
        model=model, args=sft_cfg, processing_class=tok, peft_config=peft_cfg,
        train_dataset=ds["train"], eval_dataset=ds["dev"],
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
