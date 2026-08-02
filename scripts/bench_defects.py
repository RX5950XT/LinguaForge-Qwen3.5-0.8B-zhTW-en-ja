"""下游缺陷復現 / 驗收：30 句客觀指標（移植自 VoiceInk scripts/bench-quant-quality.js）。

用途是「同一組樣本、同一組指標」的前後對照，不是翻譯品質評分。
指標全部客觀可測：專名保留、行數保留、重複迴圈、複誦、空輸出、長度比，
另加本次要修的兩類：A 類標籤前綴（tag）、C 類憑空年份（year）。

用法：
  uv run python scripts/bench_defects.py --label ship                      # beam4 出貨解碼
  uv run python scripts/bench_defects.py --label greedy --beams 1          # greedy 對照
  uv run python scripts/bench_defects.py --label v6 --adapter outputs/sft-v6
輸出：results/defects/<label>.json（含每句原始輸出，改指標不必重跑推論）
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import DECODE, INSTR, LENGTH_PENALTY, NUM_BEAMS, SYSTEM, stop_token_ids

ROOT = Path(__file__).parent.parent
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MERGED = ROOT / "release" / "merged-bf16-v5e"

# 30 句均衡樣本，與 VoiceInk scripts/bench-quant-quality.js 的 CASES 逐字對齊。
CASES = [
    # 專名／數字
    ("n1", "JPMorgan/Reuters forecast an August release.", "zhtw", ["JPMorgan"]),
    ("n2", "Im predicting it to beat Kimi k3 and maybe be around Sol level.", "zhtw", ["Kimi", "Sol"]),
    ("n3", "GLM 5.5 will launch in August with 1T+ total parameters.", "zhtw", ["GLM", "1T"]),
    ("n4", "The NVIDIA H200 has 141GB of HBM3e memory.", "zhtw", ["NVIDIA", "H200", "141"]),
    ("n5", "Anthropic released Claude Opus 4.5 on November 24, 2025.", "zhtw", ["Anthropic", "Claude"]),
    ("n6", "TSMC will start 2nm mass production in Hsinchu next year.", "zhtw", ["TSMC", "2nm"]),
    ("n7", "The flight from Taipei to Tokyo takes 3 hours and 20 minutes.", "zhtw", ["3", "20"]),
    ("n8", "Revenue grew 47% to $1.2 billion in Q3.", "zhtw", ["47", "1.2"]),
    # 散文
    ("p1", "This is an 80,000-tonne hydraulic forging press, one of the largest metal-forming machines ever built.", "zhtw", ["80,000"]),
    ("p2", "At this scale, forging is about changing the internal structure of the material itself.", "zhtw", []),
    ("p3", "A pig would break through the fence at night and wander off into the woods.", "zhtw", []),
    ("p4", "The patient should take this medication twice a day after meals.", "zhtw", []),
    ("p5", "She opened the window and listened to the sound of the rain.", "zhtw", []),
    ("p6", "We could not always tell who had already been fed and who had not.", "zhtw", []),
    # 短名詞片語
    ("s1", "Open weight release", "zhtw", []),
    ("s2", "Up to 1M context", "zhtw", ["1M"]),
    ("s3", "Free shipping on all orders", "zhtw", []),
    ("s4", "Battery life: 18 hours", "zhtw", ["18"]),
    # 多行
    ("m1", "Total parameters: 1T+\nOpen weight release\nUp to 1M context", "zhtw", ["1T", "1M"]),
    ("m2", "First, boil the water.\nSecond, add the noodles.\nThird, wait three minutes.", "zhtw", []),
    # 口語／不完整句
    ("c1", "I mean, I went to the night market yesterday and it was packed.", "zhtw", []),
    ("c2", "Sounds small-time, but at an average of $200-300 per adult pig, every loss hurts.", "zhtw", ["200"]),
    ("c3", "What are your predictions for GLM 5.5?", "zhtw", ["GLM"]),
    # zh-TW → en
    ("e1", "週末的夜市人聲鼎沸。", "en", []),
    ("e2", "請把窗戶打開，讓新鮮空氣進來。", "en", []),
    ("e3", "台積電明年將在新竹量產 2 奈米製程。", "en", ["2"]),
    ("e4", "這台機器重達八萬噸，是全球最大的金屬成形設備之一。", "en", []),
    # ja
    ("j1", "週末の夜市はとても賑やかです。", "zhtw", []),
    ("j2", "The night market is crowded on weekends.", "ja", []),
    ("j3", "明日は友達と映画を見に行きます。", "en", []),
]

# A 類：模型自行加上、原文沒有的標籤前綴。偵測用（不做剝除，剝除是下游止血）。
TAG_PATTERNS = [
    ("label", re.compile(r"^[ \t]*(?:說明|備註|註解|註|注意|提示|問|答|標題|內容|摘要|總結|結論|原文|譯者|譯文|翻譯|Note|Q|A)[：:]")),
    ("enum", re.compile(r"^[ \t]*\d{1,2}[.、)]\s")),
    ("figure", re.compile(r"^[ \t]*(?:圖\s*\d*\s*[.號：:]|圖為|照片為|圖片為)")),
    ("select", re.compile(r"^[ \t]*選擇[：:]?")),
    ("narrate", re.compile(r"^[ \t]*(?:故事說|據報導|據報道|根據報導|根據報道|報導說|報道稱)")),
]
YEAR = re.compile(r"(?:19|20)\d{2}")


def find_tag_prefix(src: str, out: str) -> str | None:
    """回傳命中的樣式名；原文首行本來就有同樣標籤時不算（避免誤傷）。"""
    for name, pat in TAG_PATTERNS:
        if pat.match(out) and not pat.match(src):
            return name
    return None


def find_repetition_loop(text: str) -> str | None:
    """與 VoiceInk findRepetitionLoop 等價：連續重複片段，或任一 6 字片段出現 ≥3 次。"""
    t = re.sub(r"\s+", "", text or "")
    if len(t) < 16:
        return None
    m = re.search(r"(.{4,24}?)\1+", t, re.DOTALL)
    if m:
        return m.group(1)
    seen: dict[str, int] = {}
    for i in range(len(t) - 5):
        seg = t[i:i + 6]
        n = seen.get(seg, 0) + 1
        if n >= 3:
            return seg
        seen[seg] = n
    return None


def defects(case, out: str) -> list[str]:
    cid, text, target, keep = case
    bad = []
    if not out:
        return ["empty"]
    bad += [f"keep:{k}" for k in keep if k not in out]
    src_lines = len(text.split("\n"))
    if src_lines > 1 and len([l for l in out.split("\n") if l.strip()]) < src_lines:
        bad.append("lines")
    if find_repetition_loop(out):
        bad.append("loop")
    if out == text.strip():
        bad.append("echo")
    tag = find_tag_prefix(text, out)
    if tag:
        bad.append(f"tag:{tag}")
    ghost = sorted(set(YEAR.findall(out)) - set(YEAR.findall(text)))
    if ghost:
        bad.append("year:" + ",".join(ghost))
    ratio = len(out) / len(text)
    if ratio < 0.3 or ratio > 3:
        bad.append(f"len:{ratio:.1f}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA 目錄；不給則用 release/merged-bf16-v5e")
    ap.add_argument("--base", action="store_true", help="用未微調的 base 對照")
    ap.add_argument("--beams", type=int, default=NUM_BEAMS)
    ap.add_argument("--length-penalty", type=float, default=LENGTH_PENALTY)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    src_id = MODEL_ID if (args.base or args.adapter) else str(MERGED)
    tok = AutoTokenizer.from_pretrained(src_id)
    model = AutoModelForImageTextToText.from_pretrained(
        src_id, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    eos_ids = stop_token_ids(tok)

    rows = []
    for case in CASES:
        cid, text, target, _ = case
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{INSTR[target]}\n{text}"}]]
        inputs = tok.apply_chat_template(convs, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt",
                                         padding=True).to("cuda")
        gen_kwargs = dict(DECODE[target])
        if args.beams > 1:
            gen_kwargs["num_beams"] = args.beams
            gen_kwargs["length_penalty"] = args.length_penalty
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, eos_token_id=eos_ids,
                                 pad_token_id=tok.pad_token_id, **gen_kwargs)
        out = tok.decode(gen[0][inputs["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        d = defects(case, out)
        rows.append({"id": cid, "target": target, "src": text, "out": out, "defects": d})
        flag = ("  ⚠ " + ",".join(d)) if d else ""
        print(f"{cid:3} [{target:4}] {out!r}{flag}", flush=True)

    summary = {
        "label": args.label,
        "beams": args.beams,
        "length_penalty": args.length_penalty if args.beams > 1 else None,
        "n": len(CASES),
        "有缺陷句": sum(1 for r in rows if r["defects"]),
        "缺陷總數": sum(len(r["defects"]) for r in rows),
        "A_標籤前綴": sum(1 for r in rows if any(x.startswith("tag") for x in r["defects"])),
        "B_專名遺失": sum(len([x for x in r["defects"] if x.startswith("keep")]) for r in rows),
        "C_憑空年份": sum(1 for r in rows if any(x.startswith("year") for x in r["defects"])),
        "D_行數遺失": sum(1 for r in rows if "lines" in r["defects"]),
        "退化迴圈": sum(1 for r in rows if "loop" in r["defects"]),
        "長度異常": sum(1 for r in rows if any(x.startswith("len") for x in r["defects"])),
    }
    keep_total = sum(len(c[3]) for c in CASES)
    summary["B_專名保留率"] = round((keep_total - summary["B_專名遺失"]) / keep_total * 100, 1)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = ROOT / "results" / "defects"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
