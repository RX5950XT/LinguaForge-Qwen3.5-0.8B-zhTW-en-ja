"""出貨硬閘：把 `CLAUDE.md` 的六條硬閘全部機器化。任一 FAIL → exit 1。

先前這支只覆蓋「洩漏 2 格 + COMET 4 格」，但 CLAUDE.md 寫的是六方向 COMET
＋ BELEBELE/知識六格＋ doc 兩條＋翻譯機率，**漏掉的正好包含 en→zhtw**
（唯一有爭議的那一格）。人工核對表格不是閘，會漏會累會忘。

三個來源、兩種比法：
- FLORES（`<tag>-flores.json`）：洩漏與 COMET，**對 baseline 動態比**
- BELEBELE / 知識（`bench/<tag>.json`）：**對 baseline 動態比**（−3.0）
- doc / 翻譯機率（`capability/<tag>.json`）：**絕對值，不比 base**
  （base 自己尾段會超譯，拿它當地板等於問「候選有沒有跟 base 一樣多話」）

COMET 的三段判定：分數是 n≈1000 段的平均，±0.4 在雜訊帶內。
落在 [base−tol, base) 判 **TIE?**，不算 FAIL 但必須跑 paired_bootstrap 補 CI；
CI 跨 0 才算平手過關，CI 整段在 0 以下就是真退步。指令由本工具印出來。

用法：
  uv run python scripts/regression_guard.py --candidate v5e
  uv run python scripts/regression_guard.py --candidate v5f --baseline base-full --bench-baseline base
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
DIRECTIONS = ["en->zhtw", "zhtw->en", "en->ja", "ja->en", "ja->zhtw", "zhtw->ja"]
LEAK_DIRECTIONS = ["en->zhtw", "ja->zhtw"]      # 只有目標側是中文才量得到洩漏
LANGS = ["zhtw", "ja", "en"]

LEAK_TOL = 0.3          # 洩漏 ≤ base + 0.3
COMET_TOL = 0.5         # COMET 雜訊帶；落帶內判 TIE? 並要求 CI
BENCH_TOL = 3.0         # BELEBELE / 知識 ≥ base − 3.0
TAIL_MIN = 0.80         # doc 尾段譯出比（絕對值）
TRUNC_MAX = 5.0         # doc 腰斬率（絕對值）
MERE_TRANS_MAX = 5.0    # 通用題被當成翻譯題作答的比率（絕對值）


def find(*patterns):
    """results/ 有分類子目錄，讀取端一律 rglob（見 results/README.md）。"""
    for p in patterns:
        f = next(RESULTS.rglob(p), None)
        if f:
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def verdict(cand, bound, mode, tol_band=None):
    """回傳 PASS / TIE? / FAIL。tol_band 有值時，落在容忍帶內判 TIE?。"""
    if mode == "max":
        if cand <= bound:
            return "PASS"
        return "FAIL"
    if cand >= bound:
        return "PASS"
    if tol_band is not None and cand >= bound - tol_band:
        return "TIE?"
    return "FAIL"


def resolve_tie(cand_tag, base_tag, direction):
    """TIE? 交給 paired bootstrap 裁決。檔名對齊 tools/comet/paired_bootstrap.out_path()。

    找不到 CI → MISSING（沒跑過的閘不算過），不是默默放行。
    """
    d = direction.replace("->", "2")
    f = RESULTS / "bootstrap" / f"{base_tag}-flores__{cand_tag}-flores__{d}.json"
    if not f.exists():
        return "MISSING", "缺 CI"
    ci = json.loads(f.read_text(encoding="utf-8"))
    if ci["crosses_zero"]:
        return "PASS", f"平手 CI[{ci['ci_lo']:+.2f},{ci['ci_hi']:+.2f}]"
    return "FAIL", f"CI[{ci['ci_lo']:+.2f},{ci['ci_hi']:+.2f}] 未跨 0"


def collect(cand_tag, base_tag, bench_base_tag):
    """把三個來源攤平成 (組別, 項目, base, cand, 界線, mode, verdict) 列。

    base 為 None 代表絕對閘。任一來源缺檔 → 該組回一列 verdict="MISSING"。
    """
    rows = []

    fl_c = find(f"{cand_tag}-flores.json")
    fl_b = find(f"{base_tag}-flores.json")
    if fl_c is None or fl_b is None:
        rows.append(("FLORES", f"缺 {cand_tag}/{base_tag}-flores.json", None, None, None, "", "MISSING"))
    else:
        c, b = fl_c["results"], fl_b["results"]
        for d in LEAK_DIRECTIONS:
            bv, cv = b.get(d, {}).get("simplified_leak_pct"), c.get(d, {}).get("simplified_leak_pct")
            if bv is None or cv is None:
                rows.append(("洩漏", d, bv, cv, None, "max", "MISSING"))
                continue
            rows.append(("洩漏", d, bv, cv, bv + LEAK_TOL, "max",
                         verdict(cv, bv + LEAK_TOL, "max")))
        for d in DIRECTIONS:
            bv, cv = b.get(d, {}).get("comet"), c.get(d, {}).get("comet")
            if bv is None or cv is None:
                rows.append(("COMET", d, bv, cv, None, "min", "MISSING"))
                continue
            rows.append(("COMET", d, bv, cv, bv, "min",
                         verdict(cv, bv, "min", COMET_TOL)))

    bn_c = find(f"bench/{cand_tag}.json")
    bn_b = find(f"bench/{bench_base_tag}.json")
    if bn_c is None or bn_b is None:
        rows.append(("通用能力", f"缺 bench/{cand_tag}|{bench_base_tag}.json", None, None, None, "", "MISSING"))
    else:
        for axis, label in (("belebele", "BELEBELE"), ("knowledge", "知識")):
            for lg in LANGS:
                bv = bn_b.get(axis, {}).get(lg, {}).get("acc")
                cv = bn_c.get(axis, {}).get(lg, {}).get("acc")
                if bv is None or cv is None:
                    rows.append((label, lg, bv, cv, None, "min", "MISSING"))
                    continue
                rows.append((label, lg, bv, cv, bv - BENCH_TOL, "min",
                             verdict(cv, bv - BENCH_TOL, "min")))

    cap = find(f"capability/{cand_tag}.json")
    if cap is None:
        rows.append(("行為", f"缺 capability/{cand_tag}.json", None, None, None, "", "MISSING"))
    else:
        doc = cap.get("doc") or {}
        for d in DIRECTIONS:
            for key, bound, mode, label in (
                ("tail_ratio_median", TAIL_MIN, "min", "doc 尾段"),
                ("truncated_pct", TRUNC_MAX, "max", "doc 腰斬%"),
            ):
                cv = doc.get(d, {}).get(key)
                rows.append((label, d, None, cv, bound, mode,
                             "MISSING" if cv is None else verdict(cv, bound, mode)))
        cv = (cap.get("general") or {}).get("mere_translation_pct")
        rows.append(("翻譯機率", "n=90", None, cv, MERE_TRANS_MAX, "max",
                     "MISSING" if cv is None else verdict(cv, MERE_TRANS_MAX, "max")))

    # TIE? 不是最終判定，交給 CI 裁決成 PASS/FAIL，缺 CI 就是 MISSING。
    # 這一步做完，本工具的輸出就不再需要任何人工補充。
    out = []
    for r in rows:
        if r[6] != "TIE?":
            out.append(r + ("",))
            continue
        vd, note = resolve_tie(cand_tag, base_tag, r[1])
        out.append(r[:6] + (vd, f"TIE? {note}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="base-full", help="FLORES 對照（唯一可用的是 base-full）")
    ap.add_argument("--bench-baseline", default="base", help="BELEBELE/知識 對照")
    args = ap.parse_args()

    rows = collect(args.candidate, args.baseline, args.bench_baseline)
    print(f"# 出貨硬閘：{args.candidate} vs {args.baseline}(FLORES) / {args.bench_baseline}(bench)\n")
    print(f"{'閘':<10} {'項目':<10} {'base':>7} {'cand':>7} {'界線':>7}  判定")
    print("-" * 70)
    f = lambda v: f"{v:7.2f}" if isinstance(v, (int, float)) else f"{'-':>7}"
    for group, item, bv, cv, bound, _mode, vd, note in rows:
        mark = " <<<" if vd == "FAIL" else ""
        tail = f"  {note}" if note else ""
        print(f"{group:<10} {item:<10} {f(bv)} {f(cv)} {f(bound)}  {vd}{mark}{tail}")

    fails = [r for r in rows if r[6] == "FAIL"]
    missing = [r for r in rows if r[6] == "MISSING"]
    print()
    need_ci = [r for r in missing if r[7].startswith("TIE?")]
    if need_ci:
        print("缺 CI — 落在雜訊帶的方向必須補跑 paired bootstrap，CI 跨 0 才算過：")
        for _g, d, *_ in need_ci:
            print(f"  uv run --project tools/comet python tools/comet/paired_bootstrap.py "
                  f"--a {args.baseline}-flores --b {args.candidate}-flores "
                  f"--direction {d.replace('->', '2')}")
        print()
    if fails:
        sys.exit(f"RESULT: FAIL ×{len(fails)} — 不得出貨")
    if missing:
        # 缺值不是 FAIL（可能只是還沒跑），但更不是 PASS —— 用第三個 exit code 分開
        print(f"RESULT: 不完整 — {len(missing)} 項缺值，沒跑過的閘不算過")
        sys.exit(2)
    tied = [r for r in rows if r[7].startswith("TIE?")]
    print(f"RESULT: PASS — 六條硬閘全過"
          + (f"（其中 {len(tied)} 格靠 CI 判平手）" if tied else ""))


if __name__ == "__main__":
    main()
