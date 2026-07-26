"""Phase 0 smoke test: 載入 Qwen3.5-0.8B、三語翻譯推論、10 步 LoRA 訓練不 OOM。"""

import gc
import sys

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"


def gb(n: int) -> str:
    return f"{n / 1024**3:.2f} GB"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    return tok, model


def translate(tok, model, instruction: str, text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": f"{instruction}\n{text}"},
    ]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    n_in = inputs["input_ids"].shape[1]
    return tok.decode(out[0][n_in:], skip_special_tokens=True).strip()


def inference_test(tok, model):
    cases = [
        ("翻譯成日文：", "今天天氣很好，我們去公園散步吧。"),
        ("Translate to Traditional Chinese (Taiwan):", "The high-speed rail connects Taipei and Kaohsiung in about 90 minutes."),
        ("英語に翻訳して：", "台湾のタピオカミルクティーは世界中で人気があります。"),
    ]
    for instr, text in cases:
        result = translate(tok, model, instr, text)
        print(f"[{instr}] {text}\n  -> {result}\n")
        assert len(result) > 0, "empty generation"
    print(f"inference peak VRAM: {gb(torch.cuda.max_memory_allocated())}")


def list_linear_modules(model):
    names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            names.add(name.rsplit(".", 1)[-1])
    print("linear module leaf names:", sorted(names))
    top = sorted({n.split(".")[0] for n, _ in model.named_modules() if n})
    print("top-level modules:", top)


def train_test(tok, model):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = [
        {"messages": [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": "翻譯成英文：你好，世界。"},
            {"role": "assistant", "content": "Hello, world."},
        ]}
    ] * 64
    ds = Dataset.from_list(rows)

    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                        "out_proj"],
    )
    cfg = SFTConfig(
        output_dir="outputs/smoke", max_steps=10, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=1e-4, bf16=True,
        gradient_checkpointing=True, optim="adamw_torch", logging_steps=1,
        max_length=512, report_to=[], save_strategy="no",
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    print(f"train peak VRAM: {gb(torch.cuda.max_memory_allocated())}")


if __name__ == "__main__":
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    assert torch.cuda.is_available()
    tok, model = load_model()
    print(f"model loaded, params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    list_linear_modules(model)
    inference_test(tok, model)
    if "--train" in sys.argv:
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        train_test(tok, model)
    print("SMOKE TEST PASSED")
