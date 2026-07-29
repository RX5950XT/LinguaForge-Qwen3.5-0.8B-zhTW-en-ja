"""Phase 2/4/v3: 多基準六方向翻譯評測 — chrF++ / BLEU / 簡體洩漏率。

支援可插拔基準（--benchmark）：
  flores  FLORES-200 devtest（維基書面體，zho_Hant 原生繁體）
  ntrex   NTREX-128（新聞，zho_Hant 原生繁體 + jpn，English-pivot 多平行，全六方向）
  wmt22   WMT22 General-MT（多領域，sacrebleu；僅 4 個英配方向，zh 簡體→s2twp）
  alt     ALT（Wikinews，多平行，全六方向，zh 簡體→s2twp）
  tico19  TICO-19（醫療，僅 en↔zhtw，zh 簡體→s2twp）

各基準各自寫 results/<tag>-<benchmark>.json 與 results/hyp/<tag>-<benchmark>/<dir>.{src,ref,hyp}.txt
（COMET 由 tools/comet/score.py、CometKiwi 由 tools/comet/kiwi.py 讀 hyp 檔另算，環境隔離）。

參考策略：FLORES/NTREX zho_Hant 原生繁體 → reference-based 乾淨可信；
WMT/ALT/TICO 為簡體 → 對 zh 端做 s2twp（reference-based 分數次要/方向性，主看 CometKiwi+洩漏率）。
絕不 s2twp 模型自己的輸出（會藏住洩漏）。

用法：
  uv run python scripts/evaluate.py --tag v2 --benchmark ntrex --adapter outputs/sft-v2 --full
  uv run python scripts/evaluate.py --tag v2 --benchmark all   --adapter outputs/sft-v2 --full
  uv run python scripts/evaluate.py --tag v2 --benchmark flores --adapter outputs/sft-v2 --beams 4
"""

import argparse
import json
from pathlib import Path

import torch
from opencc import OpenCC
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).parent.parent
MODEL_ID = "Qwen/Qwen3.5-0.8B"

# FLORES-200 / NTREX 共用的三語代碼（NTREX config 名稱與 FLORES 一致）
LANGS = {"en": "eng_Latn", "ja": "jpn_Jpan", "zhtw": "zho_Hant"}
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
DIRECTIONS = [("en", "zhtw"), ("zhtw", "en"), ("en", "ja"),
              ("ja", "en"), ("ja", "zhtw"), ("zhtw", "ja")]
INSTR = {"zhtw": "翻譯成繁體中文：", "en": "翻譯成英文：", "ja": "翻譯成日文："}
SYSTEM = "You are a professional translator."

cc_s2tw = OpenCC("s2tw")
cc_s2twp = OpenCC("s2twp")

# 洩漏偵測不能用「轉換器會不會改寫這個字」來判定——那是異體字偏好，不是簡繁。
# 舊版拿 s2tw 整句 round-trip，實測 86 種被判洩漏的字裡前 11 名（群 里 吃 游 托 岩
# 台 斗 峰 床 征，佔 1127 次中的 878 次）全是台灣標準用字，被要求改成羣 裏 喫 遊
# 託 巖 臺 鬥 峯 牀 徵 這些台灣不用的異體字，洩漏率因此灌水約 5~7 倍。
# 改成：逐字（避開 s2t 的詞組規則）判「這個字是不是簡體專用」＝ 它已是簡化形
# （t2s 不動它）且有對應的正體形（s2t 會改它），再扣掉下面這批簡化時被合併、
# 但本身就是台灣正字的字。名單由 41,052 句訓練 zhtw 目標＋FLORES zh-TW 參考
# 統計而來：出現 ≥10 次者為台灣用字，1~3 次者才是真殘留（分界乾淨）。
cc_t2s = OpenCC("t2s")
cc_s2t = OpenCC("s2t")
TW_VARIANTS = frozenset("群里吃秘床峰游托准干伙岩后征采痴斗台杰划岳皂唇占雇佣栗筑灶丑朴")
_SIMPLIFIED = {c for c in {chr(i) for i in range(0x4E00, 0xA000)}
               if c not in TW_VARIANTS
               and cc_t2s.convert(c) == c and cc_s2t.convert(c) != c}


def _truncate(texts, limit):
    return {k: (v[:limit] if limit else v) for k, v in texts.items()}


def _multiparallel_pairs(texts, simplified_zh):
    """{lang: [sents]}（同序多平行）→ {(src,tgt): (src_sents, ref_sents)}。

    simplified_zh=True 時把 zh 端（不論來源或目標）s2twp 統一成台灣正體。
    """
    n = {len(v) for v in texts.values()}
    assert len(n) == 1, f"unaligned splits: { {k: len(v) for k, v in texts.items()} }"
    out = {}
    for src, tgt in DIRECTIONS:
        if src not in texts or tgt not in texts:
            continue
        s, r = list(texts[src]), list(texts[tgt])
        if simplified_zh:
            if src == "zhtw":
                s = [cc_s2twp.convert(x) for x in s]
            if tgt == "zhtw":
                r = [cc_s2twp.convert(x) for x in r]
        out[(src, tgt)] = (s, r)
    return out


def load_flores(limit):
    """Meta 官方 FLORES-200 tarball（公開），快取於 data/flores200；zho_Hant 原生繁體。"""
    import tarfile
    import urllib.request

    base = ROOT / "data" / "flores200"
    devtest = base / "flores200_dataset" / "devtest"
    if not devtest.exists():
        base.mkdir(parents=True, exist_ok=True)
        tgz = base / "flores200_dataset.tar.gz"
        print(f"  downloading {FLORES_URL}")
        urllib.request.urlretrieve(FLORES_URL, tgz)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(base, filter="data")
        tgz.unlink()
    texts = {lang: (devtest / f"{code}.devtest").read_text(encoding="utf-8").rstrip("\n").split("\n")
             for lang, code in LANGS.items()}
    return _multiparallel_pairs(_truncate(texts, limit), simplified_zh=False)


def load_ntrex(limit):
    """NTREX-128：各語言 config 單欄 text、test 1997 行、逐行對齊；zho_Hant 原生繁體。"""
    from datasets import load_dataset

    texts = {}
    for lang, code in LANGS.items():
        ds = load_dataset("davidstap/NTREX", code, split="test")
        texts[lang] = [r["text"] for r in ds]
    return _multiparallel_pairs(_truncate(texts, limit), simplified_zh=False)


def load_alt(limit):
    """ALT（Wikinews 多平行）：每列 translation dict 含 en/ja/zh（簡體）→ s2twp。"""
    from datasets import load_dataset

    ds = load_dataset("mutiyama/alt", "alt-parallel", split="test")
    texts = {"en": [], "ja": [], "zhtw": []}
    for row in ds:
        t = row["translation"]
        en, ja, zh = t.get("en"), t.get("ja"), t.get("zh")
        if en and ja and zh:
            texts["en"].append(en)
            texts["ja"].append(ja)
            texts["zhtw"].append(zh)
    return _multiparallel_pairs(_truncate(texts, limit), simplified_zh=True)


def load_tico19(limit):
    """TICO-19（醫療，僅 en-zh）：sourceString=en / targetString=zh（簡體）→ s2twp。

    gmnlp/tico19 是 script-based（新版 datasets 拒絕）→ 直接讀自動轉出的 parquet。
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    pq = hf_hub_download("gmnlp/tico19", "en-zh/test/0000.parquet",
                         repo_type="dataset", revision="refs/convert/parquet")
    ds = load_dataset("parquet", data_files=pq, split="train")
    texts = {"en": [r["sourceString"] for r in ds],
             "zhtw": [r["targetString"] for r in ds]}
    return _multiparallel_pairs(_truncate(texts, limit), simplified_zh=True)


def load_wmt22(limit):
    """WMT22 General-MT（sacrebleu）：雙語 per-pair，僅 4 個英配方向；zh 簡體→s2twp。"""
    import sacrebleu

    mapping = {("en", "zhtw"): "en-zh", ("zhtw", "en"): "zh-en",
               ("en", "ja"): "en-ja", ("ja", "en"): "ja-en"}
    pairs = {}
    for (src, tgt), lp in mapping.items():
        try:
            src_sents = Path(sacrebleu.get_source_file("wmt22", lp)).read_text(
                encoding="utf-8").rstrip("\n").split("\n")
            ref_sents = Path(sacrebleu.get_reference_files("wmt22", lp)[0]).read_text(
                encoding="utf-8").rstrip("\n").split("\n")
        except Exception as e:
            print(f"  !! wmt22 {lp} unavailable ({e}), skipped")
            continue
        if limit:
            src_sents, ref_sents = src_sents[:limit], ref_sents[:limit]
        if src == "zhtw":
            src_sents = [cc_s2twp.convert(x) for x in src_sents]
        if tgt == "zhtw":
            ref_sents = [cc_s2twp.convert(x) for x in ref_sents]
        pairs[(src, tgt)] = (src_sents, ref_sents)
    return pairs


BENCHMARKS = {"flores": load_flores, "ntrex": load_ntrex, "wmt22": load_wmt22,
              "alt": load_alt, "tico19": load_tico19}


def stop_token_ids(tok) -> list[int]:
    """SFT 資料以 <|im_end|> 收尾，但 base config 的 eos 是 <|endoftext|>。
    兩者都當 eos，否則微調後模型會在正確譯文後失控重複到 max_new_tokens。"""
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(ids)


# 出貨解碼預設，依「目標語言」分流。實測見 docs/RESEARCH-v5.md F3（同 12 篇文件）：
#   ja / en 輸出：greedy 會陷入 75~100% 貪婪迴圈（base 自帶的收尾失敗，訓練會放大），
#                 rep 1.1 把 en→ja 從 7.28 拉到 17.37、破 base 的 8.83 近兩倍。
#   zh-TW 輸出：rep 1.1 壓掉重複字元後模型改挑簡體變體，簡體洩漏 en→zhtw 4.65%→13.06%、
#               ja→zhtw 0.56%→3.85%，只換到 +1.63 / +3.08 chrF++——不划算，維持 greedy。
# CLI 的 --rep-penalty 仍可整體覆寫（做對照實驗用）。
DECODE = {"ja": {"repetition_penalty": 1.1},
          "en": {"repetition_penalty": 1.1},
          "zhtw": {}}


def batched_translate(tok, model, prompts, batch_size, gen_kwargs) -> list[str]:
    eos_ids = stop_token_ids(tok)
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": p}] for p in chunk]
        inputs = tok.apply_chat_template(
            convs, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 eos_token_id=eos_ids, pad_token_id=tok.pad_token_id,
                                 **gen_kwargs)
        n_in = inputs["input_ids"].shape[1]
        outs.extend(tok.decode(g[n_in:], skip_special_tokens=True).strip() for g in gen)
        if (i // batch_size) % 5 == 0:
            print(f"    {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def score(direction, hyps, refs):
    import sacrebleu

    tgt = direction[1]
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    bleu_tok = {"zhtw": "zh", "ja": "ja-mecab", "en": "13a"}[tgt]
    try:
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=bleu_tok).score
    except Exception as e:
        print(f"    BLEU failed ({e}), skipping")
        bleu = None
    leak = None
    if tgt == "zhtw":
        leak = sum(any(c in _SIMPLIFIED for c in h) for h in hyps) / len(hyps) * 100
    return {"chrf++": round(chrf, 2), "bleu": round(bleu, 2) if bleu else None,
            "simplified_leak_pct": round(leak, 2) if leak is not None else None}


def run_benchmark(bench, tok, model, limit, batch, gen_kwargs, tag, meta):
    print(f"== loading benchmark: {bench} ==")
    pairs = BENCHMARKS[bench](limit)
    if not pairs:
        print(f"  !! {bench} produced no directions, skipped")
        return
    eff_tag = f"{tag}-{bench}"
    hyp_dir = ROOT / "results" / "hyp" / eff_tag
    hyp_dir.mkdir(parents=True, exist_ok=True)
    results, n = {}, 0
    for (src_l, tgt_l), (src_sents, refs) in pairs.items():
        name = f"{src_l}->{tgt_l}"
        n = len(src_sents)
        print(f"== [{bench}] {name} ({n}) ==")
        prompts = [f"{INSTR[tgt_l]}\n{s}" for s in src_sents]
        hyps = batched_translate(tok, model, prompts, batch,
                                 {**DECODE[tgt_l], **gen_kwargs})
        results[name] = score((src_l, tgt_l), hyps, refs)
        print(f"  {results[name]}")
        stem = name.replace("->", "2")
        for suffix, rows in (("src", src_sents), ("ref", refs), ("hyp", hyps)):
            with open(hyp_dir / f"{stem}.{suffix}.txt", "w", encoding="utf-8",
                      newline="\n") as f:
                f.write("\n".join(r.replace("\n", " ") for r in rows) + "\n")
    out = ROOT / "results" / f"{eff_tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tag": eff_tag, "benchmark": bench, "model": meta["model"],
                   "adapter": meta["adapter"], "n": n,
                   "gen": {k: v for k, v in gen_kwargs.items()},
                   "decode_defaults": DECODE,   # 依目標語言分流，CLI 可覆寫
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"results -> {out}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="結果標籤，如 v2 / v3（實際檔名 <tag>-<benchmark>）")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 路徑")
    ap.add_argument("--model", default=MODEL_ID, help="模型 ID 或 merged 路徑")
    ap.add_argument("--benchmark", default="flores",
                    choices=[*BENCHMARKS, "all"], help="評測基準")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--full", action="store_true", help="不截斷（用整份基準）")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--beams", type=int, default=1, help=">1 開 beam search")
    ap.add_argument("--rep-penalty", type=float, default=None,
                    help="覆寫 DECODE 的每目標語言預設；給 1.0 即可跑純 greedy 對照")
    ap.add_argument("--length-penalty", type=float, default=None)
    # rep-penalty 對 zh-TW 是禁藥（F3：重新加權已出現的 token，會把繁體字推成簡體變體，
    # en→zhtw 洩漏 4.65%→13.06%）。no_repeat_ngram_size 是硬性 n-gram 封鎖，
    # 不動單字機率分布，所以不會有那個副作用——用它處理 F11 的重複尾巴。
    ap.add_argument("--no-repeat-ngram", type=int, default=None,
                    help="禁止重複的 n-gram 長度（zh-TW 可用，不像 rep-penalty）")
    ap.add_argument("--nf4", action="store_true",
                    help="4-bit NF4 載入 base（配 QLoRA adapter，復現訓練精度）")
    ap.add_argument("--int8", action="store_true",
                    help="8-bit LLM.int8() 載入 base（量化稅對照）")
    args = ap.parse_args()
    limit = None if args.full else args.limit

    gen_kwargs = {}
    if args.beams > 1:
        gen_kwargs["num_beams"] = args.beams
    if args.rep_penalty:
        gen_kwargs["repetition_penalty"] = args.rep_penalty
    if args.length_penalty is not None:
        gen_kwargs["length_penalty"] = args.length_penalty
    if args.no_repeat_ngram:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram

    print("== loading model ==")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    quantized = args.nf4 or args.int8
    if args.nf4:  # 復現 QLoRA 訓練精度：NF4 4-bit base（見 train_sft.py）
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        load_kwargs["device_map"] = {"": 0}
    elif args.int8:  # LLM.int8() 8-bit：量化稅對照
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    if not quantized:
        model = model.cuda()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  adapter loaded: {args.adapter}")
    model.eval()

    meta = {"model": args.model, "adapter": args.adapter}
    benches = list(BENCHMARKS) if args.benchmark == "all" else [args.benchmark]
    for bench in benches:
        run_benchmark(bench, tok, model, limit, args.batch, gen_kwargs, args.tag, meta)


if __name__ == "__main__":
    main()
