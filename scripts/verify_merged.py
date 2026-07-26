"""載入 merged 模型跑六方向各一句，結果寫 UTF-8 檔（繞過 Windows console 編碼）。"""

from pathlib import Path

import torch
from opencc import OpenCC
from transformers import AutoModelForImageTextToText, AutoTokenizer

cc_s2t = OpenCC("s2t")
OUT = Path(__file__).parent.parent / "results" / "merged_check.txt"
CASES = [
    ("翻譯成繁體中文：", "The night market is crowded on weekends."),
    ("Translate to Traditional Chinese (Taiwan):", "I bought a new bicycle yesterday."),
    ("翻譯成日文：", "這家拉麵店的湯頭非常濃郁。"),
    ("翻譯成英文：", "台北的捷運系統又快又乾淨。"),
    ("翻譯成繁體中文：", "彼は毎朝コーヒーを飲みます。"),
    ("翻譯成日文：", "我下週要去日本出差。"),
]

tok = AutoTokenizer.from_pretrained("outputs/merged")
model = AutoModelForImageTextToText.from_pretrained(
    "outputs/merged", dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

lines = []
for instr, text in CASES:
    msgs = [{"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": f"{instr}\n{text}"}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    res = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    leak = "简体!" if cc_s2t.convert(res) != res and "繁體" in instr + "Traditional" else ""
    lines.append(f"[{instr}] {text}\n  -> {res}  {leak}")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {OUT}")
