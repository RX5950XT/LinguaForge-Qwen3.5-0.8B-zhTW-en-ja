"""能力面板評測 — 補上舊 evaluate.py 測不到的三個維度。

舊 evaluate.py 的五個基準全是「單句翻譯」，v2/v3 在上面 COMET 84+ 但實際上：
長篇只翻前兩段就停、任何非翻譯指令都被當成待翻譯文本。本腳本就是為了讓這種
退化在數字上現形。

三軸（--axis）：
  doc      文件級翻譯（WMT24++ 按 document_id 聚合，六方向 en-pivot）
           → chrF++ / 完整度 / 漏譯 / 重複 / 簡體洩漏
  ifeval   可驗證指令跟隨（自建 zh-TW / ja / en，規則判分，IFEval 作法）
           → pass rate
  general  通用能力保留（唯一答案問答 + 「退化成翻譯機」自動偵測）
           → 正確率 / 翻譯機率

用法：
  uv run python scripts/eval_capability.py --tag base
  uv run python scripts/eval_capability.py --tag v3 --adapter outputs/sft-v3
  uv run python scripts/eval_capability.py --tag v3 --adapter outputs/sft-v3 --axis doc --docs 30
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from opencc import OpenCC
from transformers import AutoModelForImageTextToText, AutoTokenizer

from evaluate import DECODE  # 出貨解碼預設（依目標語言分流），單一真相來源

ROOT = Path(__file__).parent.parent
MODEL_ID = "Qwen/Qwen3.5-0.8B"
SYSTEM = "You are a professional translator."
INSTR = {"zhtw": "翻譯成繁體中文：", "en": "翻譯成英文：", "ja": "翻譯成日文："}
DIRECTIONS = [("en", "zhtw"), ("zhtw", "en"), ("en", "ja"),
              ("ja", "en"), ("ja", "zhtw"), ("zhtw", "ja")]
WMT24PP = {"zhtw": "en-zh_TW", "ja": "en-ja_JP"}

# s2tw 而非 s2t：s2t 把「剛才→剛纔」「人群→人羣」等正確台灣用字誤判為簡體
cc_s2tw = OpenCC("s2tw")


# ---------------------------------------------------------------- 軸 A：文件級翻譯

def load_docs(n_docs):
    """WMT24++ en→zh_TW / en→ja 共用同一批英文原文 → 按 (document_id, segment_id)
    對齊即得六方向多平行文件。回傳 {(src,tgt): [(src_doc, ref_doc), ...]}。"""
    from datasets import load_dataset

    by_lang = {}
    for lang, config in WMT24PP.items():
        ds = load_dataset("google/wmt24pp", config, split="train")
        by_lang[lang] = {(r["document_id"], r["segment_id"]): r for r in ds
                         if not r["is_bad_source"]}
    keys = sorted(set(by_lang["zhtw"]) & set(by_lang["ja"]),
                  key=lambda k: (k[0], k[1]))
    assert keys, "wmt24pp zh_TW / ja 對不齊"

    docs = {}  # document_id -> {lang: [segments]}
    for doc_id, seg_id in keys:
        d = docs.setdefault(doc_id, {"en": [], "zhtw": [], "ja": []})
        d["en"].append(by_lang["zhtw"][(doc_id, seg_id)]["source"])
        d["zhtw"].append(by_lang["zhtw"][(doc_id, seg_id)]["target"])
        d["ja"].append(by_lang["ja"][(doc_id, seg_id)]["target"])
    # 只取多句文件（單句文件測不出長篇能力）
    picked = [d for d in docs.values() if len(d["en"]) >= 3][:n_docs]
    print(f"  {len(picked)} 篇文件，平均 {sum(len(d['en']) for d in picked)/len(picked):.1f} 句")

    out = {}
    for src, tgt in DIRECTIONS:
        out[(src, tgt)] = [("\n".join(d[src]), "\n".join(d[tgt])) for d in picked]
    return out


def repeat_ratio(text, n=8):
    """字元 n-gram 重複率 — 抓「同一句無限複述」的退化模式。"""
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    c = Counter(grams)
    return sum(v - 1 for v in c.values()) / len(grams)


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")


def alignment(hyps, refs):
    """整篇字元長度比會被「多吐垃圾行」補償掉「尾段腰斬」，兩者相抵拿到假的高分
    （base 實測 line_ratio 1.4~1.5、tail 0.52~0.62，整篇比卻是 ~1.0）。
    拆成兩個指標各自看：行數對齊，以及後三分之一的譯出比例。"""
    line_ratios, tails = [], []
    for h, r in zip(hyps, refs):
        H, R = h.split("\n"), r.split("\n")
        line_ratios.append(len(H) / max(len(R), 1))
        if len(R) < 3:
            continue
        lo = 2 * len(R) // 3
        rc = sum(len(x) for x in R[lo:])
        if rc:
            tails.append(sum(len(x) for x in H[lo:]) / rc)
    return median(line_ratios), median(tails)


def score_doc(direction, srcs, hyps, refs):
    import sacrebleu

    tgt = direction[1]
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    # 完整度：譯文長度 / 參考譯文長度。<1 代表漏譯，>>1 代表重複膨脹
    ratios = [len(h) / max(len(r), 1) for h, r in zip(hyps, refs)]
    line_ratio, tail_ratio = alignment(hyps, refs)
    ratios.sort()
    trunc = sum(r < 0.5 for r in ratios) / len(ratios) * 100   # 腰斬視為漏譯
    bloat = sum(r > 2.0 for r in ratios) / len(ratios) * 100   # 膨脹視為重複
    rep = sum(repeat_ratio(h) > 0.3 for h in hyps) / len(hyps) * 100
    # 逐「行」而非逐「文件」算洩漏：文件級一篇十幾句，只要一個簡體字整篇就算洩漏，
    # 分母太粗會直接飽和到 100%，也無法跟單句基準的洩漏率比較
    lines = [x for h in hyps for x in h.split("\n") if x.strip()]
    leak = (sum(cc_s2tw.convert(x) != x for x in lines) / len(lines) * 100
            if tgt == "zhtw" and lines else None)
    return {"chrf++": round(chrf, 2),
            "completeness_median": round(ratios[len(ratios) // 2], 3),
            "line_ratio_median": round(line_ratio, 3),
            "tail_ratio_median": round(tail_ratio, 3),
            "truncated_pct": round(trunc, 1),
            "bloated_pct": round(bloat, 1),
            "repetitive_pct": round(rep, 1),
            "simplified_leak_pct": round(leak, 2) if leak is not None else None}


# ------------------------------------------------- 軸 B：可驗證指令跟隨（IFEval 作法）

def _sep_only(text, sep, n):
    """恰好 n 項、只用 sep 分隔、沒有多餘說明文字。

    只數項數會漏掉「當然可以！以下是三種水果：芒果、香蕉、鳳梨」這種帶前言的答案
    ——它照樣切出 3 項。冒號／驚嘆號／句中句號一律視為多餘文字。"""
    t = text.strip().rstrip("。.")
    if re.search(r"[：:！!。]", t):
        return False
    parts = [p for p in t.split(sep) if p.strip()]
    return len(parts) == n and "\n" not in t and len(t) < 40


def _lines(t):
    """非空行清單（模型常多吐一個換行，不該因此判錯）。"""
    return [x.strip() for x in t.strip().split("\n") if x.strip()]


def _exact(t, *ok):
    """去掉前後空白與句末標點後完全相符。"""
    return t.strip().rstrip("。.！!") in ok


def _nchar(t):
    """CJK 計字數：空白與換行不算。"""
    return len(re.sub(r"\s", "", t.strip()))


def _trad(t):
    return cc_s2tw.convert(t) == t


def _kana(t):
    return re.search(r"[぀-ヿ]", t) is not None


# 目標語言守衛：中日文題目若拿英文拒答（"I cannot answer that."）去判，
# 「不可出現 X」「N 字以內」這類約束會 vacuously 通過。自檢在檔尾 _selfcheck()。
_HAN_RE = re.compile(r"[一-鿿]")


def _han(t):
    return _HAN_RE.search(t) is not None


def _num(t):
    """數值比對前正規化：全形數字轉半形、拿掉千分位逗號與空白，
    否則「1,000」「１０００」都會被判成答錯。"""
    return (t.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
            .replace(",", "").replace("，", "").replace(" ", ""))


# 每語言 30 題。v4 那份只有 zhtw 6 / ja 5 / en 6——一題就是 16.7 個百分點，
# 「zhtw 從 66.7 退到 50.0」其實是 4/6→3/6 差一題，撐不起任何結論（見 RESEARCH-v5.md F4）。
# 全部維持「規則可判分、答案唯一」，不引入 LLM-as-judge。
# 注意每條否定型約束（不可出現 X）都要配長度下限，否則空字串會 vacuously 通過。
IFEVAL = [
    # (語言, 指令, 判分函數)
    # ---------------------------------------------------------------- zh-TW
    ("zhtw", "列出三種台灣常見的水果，只要名稱，用頓號分隔，不要任何其他文字。",
     lambda t: _sep_only(t, "、", 3)),
    ("zhtw", "用剛好兩句話描述下雨的傍晚，每句都要以「。」結尾。",
     lambda t: t.strip().count("。") == 2 and t.strip().endswith("。")),
    ("zhtw", "回答「台北」這兩個字，不要標點，不要其他任何文字。",
     lambda t: _exact(t, "台北", "臺北")),
    ("zhtw", "寫一段關於咖啡的介紹，全文不可以出現「咖啡」這兩個字。",
     lambda t: "咖啡" not in t and _nchar(t) > 20),
    ("zhtw", "用繁體中文回答：什麼是機器學習？回答不超過 50 個字。",
     lambda t: 5 < _nchar(t) <= 50 and _trad(t) and _han(t)),
    ("zhtw", "把下面三個詞按筆畫由少到多排序，用箭頭 -> 連接，不要其他文字：山、樹、木",
     lambda t: t.strip().count("->") == 2 and "\n" not in t.strip()),
    ("zhtw", "3 加 4 等於多少？只回答一個阿拉伯數字，不要其他文字。",
     lambda t: _exact(t, "7")),
    ("zhtw", "列出三個台灣的縣市，每行一個，不要編號，不要其他文字。",
     lambda t: len(_lines(t)) == 3 and all(_nchar(x) <= 6 for x in _lines(t))),
    ("zhtw", "寫一句話描述大海，全文不可以使用任何標點符號。",
     lambda t: not re.search(r"[，。、！？；：,.!?;:]", t) and _nchar(t) > 5),
    ("zhtw", "回答必須以「答案是」這三個字開頭：一年有幾個月？",
     lambda t: t.strip().startswith("答案是") and ("12" in t or "十二" in t)),
    ("zhtw", "寫出 1 到 5 這五個數字，用半形逗號分隔，不要空格，不要其他文字。",
     lambda t: _exact(t.replace(" ", ""), "1,2,3,4,5")),
    ("zhtw", "只回答 JSON，不要 markdown 圍欄也不要說明：一個物件，"
             "包含 \"city\" 與 \"country\" 兩個鍵，內容是台北。",
     lambda t: _is_json_with(t, {"city", "country"})),
    ("zhtw", "寫一段介紹春天的文字，全文不可以出現任何阿拉伯數字。",
     lambda t: not re.search(r"\d", t) and _nchar(t) > 20),
    ("zhtw", "用剛好十個字描述夏天，不要標點符號。",
     lambda t: _nchar(re.sub(r"[，。、！？；：]", "", t)) == 10),
    ("zhtw", "把「你好」重複三次，用頓號分隔，不要其他文字。",
     lambda t: _sep_only(t, "、", 3) and t.count("你好") == 3),
    ("zhtw", "地球比月球大嗎？只回答「是」或「否」一個字，不要其他文字。",
     lambda t: _exact(t, "是")),
    ("zhtw", "寫一句話介紹台灣，全文不可以出現「的」這個字。",
     lambda t: "的" not in t and _nchar(t) > 10 and _han(t)),
    ("zhtw", "寫一句話，必須以「台灣」開頭、以「島」字結尾。",
     lambda t: t.strip().startswith(("台灣", "臺灣"))
     and t.strip().rstrip("。！").endswith("島")),
    ("zhtw", "用一個問句回答：我該學哪一種程式語言？（你的回答必須以問號結尾）",
     lambda t: t.strip().endswith(("？", "?")) and _nchar(t) > 5),
    ("zhtw", "列出三個顏色，用全形逗號「，」分隔，不要頓號，不要其他文字。",
     lambda t: _sep_only(t, "，", 3) and "、" not in t),
    ("zhtw", "回答不超過 20 個字，而且必須包含「因為」兩個字：為什麼天空是藍的？",
     lambda t: "因為" in t and _nchar(t) <= 20),
    ("zhtw", "寫出四個四字成語，每行一個，不要編號，不要解釋。",
     lambda t: len(_lines(t)) == 4 and all(_nchar(x) == 4 for x in _lines(t))),
    ("zhtw", "用條列式列出三項運動，每行都要以「- 」開頭。",
     lambda t: len(_lines(t)) == 3 and all(x.startswith("- ") for x in _lines(t))),
    ("zhtw", "回答一個英文單字，全部大寫，不要其他文字：台灣的英文國名縮寫常寫作什麼？",
     lambda t: _exact(t, "TAIWAN", "ROC", "TW")),
    ("zhtw", "寫兩句話介紹夜市，第一句必須以「夜市」開頭。",
     lambda t: t.strip().count("。") == 2 and t.strip().startswith("夜市")),
    ("zhtw", "把這句話改寫成疑問句，只輸出改寫結果：他今天有來。",
     lambda t: t.strip().endswith(("？", "?")) and "來" in t and _nchar(t) < 20),
    ("zhtw", "用繁體中文寫一段關於颱風的說明，不可以出現任何簡體字，長度 30 到 80 字。",
     lambda t: 30 <= _nchar(t) <= 80 and _trad(t)),
    ("zhtw", "回答時每個字之間都要空一個半形空格：你好嗎",
     lambda t: t.strip().replace(" ", "") == "你好嗎" and t.strip().count(" ") == 2),
    ("zhtw", "列出五個數字，用頓號分隔，每個都必須是偶數，不要其他文字。",
     lambda t: _sep_only(t, "、", 5)
     and all(x.strip().isdigit() and int(x) % 2 == 0
             for x in t.strip().rstrip("。").split("、"))),
    ("zhtw", "只輸出一行，不可以換行：用一句話說明什麼是雲端運算。",
     lambda t: "\n" not in t.strip() and _nchar(t) > 10 and _han(t)),

    # ------------------------------------------------------------------- ja
    ("ja", "日本の都道府県を3つ、読点（、）で区切って名前だけ答えてください。他の文字は不要です。",
     lambda t: _sep_only(t, "、", 3)),
    ("ja", "「はい」だけを答えてください。他の文字は一切書かないでください。",
     lambda t: _exact(t, "はい")),
    ("ja", "雨の夕方を、ちょうど2つの文で描写してください。各文は「。」で終わること。",
     lambda t: t.strip().count("。") == 2 and t.strip().endswith("。")),
    ("ja", "機械学習とは何ですか。日本語で、50文字以内で答えてください。",
     lambda t: 5 < _nchar(t) <= 50 and _kana(t)),
    ("ja", "コーヒーについて説明してください。ただし「コーヒー」という単語は使わないでください。",
     lambda t: "コーヒー" not in t and _nchar(t) > 20),
    ("ja", "3たす4はいくつですか。半角数字ひとつだけで答えてください。",
     lambda t: _exact(t, "7")),
    ("ja", "日本の有名な山を3つ、1行に1つずつ書いてください。番号や説明は不要です。",
     lambda t: len(_lines(t)) == 3 and all(_nchar(x) <= 8 for x in _lines(t))),
    ("ja", "海について一文で書いてください。句読点は一切使わないでください。",
     lambda t: not re.search(r"[、。！？，．]", t) and _nchar(t) > 5 and _kana(t)),
    ("ja", "「答えは」で始めて答えてください：1年は何ヶ月ですか。",
     lambda t: t.strip().startswith("答えは") and ("12" in t or "十二" in t)),
    ("ja", "1から5までの数字を半角カンマ区切りで、空白なしで書いてください。他の文字は不要です。",
     lambda t: _exact(t.replace(" ", ""), "1,2,3,4,5")),
    ("ja", "JSONだけを出力してください。マークダウンの囲みも説明も不要です。"
           "東京について \"city\" と \"country\" の2つのキーを持つオブジェクト。",
     lambda t: _is_json_with(t, {"city", "country"})),
    ("ja", "春について書いてください。半角・全角を問わず数字は一切使わないでください。",
     lambda t: not re.search(r"[0-9０-９]", t) and _nchar(t) > 20),
    ("ja", "「こんにちは」を3回、読点（、）で区切って書いてください。他の文字は不要です。",
     lambda t: _sep_only(t, "、", 3) and t.count("こんにちは") == 3),
    ("ja", "地球は月より大きいですか。「はい」か「いいえ」のどちらか一語だけで答えてください。",
     lambda t: _exact(t, "はい")),
    ("ja", "「東京」で始まり「です」で終わる一文を書いてください。",
     lambda t: t.strip().startswith("東京")
     and t.strip().rstrip("。").endswith("です")),
    ("ja", "疑問文で答えてください（「？」で終わること）：どのプログラミング言語を学ぶべきですか。",
     lambda t: t.strip().endswith(("？", "?")) and _nchar(t) > 5),
    ("ja", "20文字以内で、かつ「なぜなら」を必ず含めて答えてください：空はなぜ青いのですか。",
     lambda t: "なぜなら" in t and _nchar(t) <= 20),
    ("ja", "四字熟語を4つ、1行に1つずつ書いてください。番号も説明も不要です。",
     lambda t: len(_lines(t)) == 4 and all(_nchar(x) == 4 for x in _lines(t))),
    ("ja", "スポーツを3つ、各行を「- 」で始める箇条書きにしてください。",
     lambda t: len(_lines(t)) == 3 and all(x.startswith("- ") for x in _lines(t))),
    ("ja", "全部大文字の英単語ひとつだけで答えてください：日本の英語名は？",
     lambda t: _exact(t, "JAPAN")),
    ("ja", "夜市について2文で書いてください。1文目は「夜市」で始めること。",
     lambda t: t.strip().count("。") == 2 and t.strip().startswith("夜市")),
    ("ja", "次の文を疑問文に書き換えて、結果だけを出力してください：彼は今日来ました。",
     lambda t: t.strip().endswith(("か。", "か", "？", "?")) and _nchar(t) < 20),
    ("ja", "台風について30文字以上80文字以内で説明してください。",
     lambda t: 30 <= _nchar(t) <= 80 and _kana(t)),
    ("ja", "偶数を5つ、読点（、）で区切って書いてください。他の文字は不要です。",
     lambda t: _sep_only(t, "、", 5)
     and all(x.strip().isdigit() and int(x) % 2 == 0
             for x in t.strip().rstrip("。").split("、"))),
    ("ja", "改行せず1行だけで答えてください：クラウドコンピューティングとは何ですか。",
     lambda t: "\n" not in t.strip() and _nchar(t) > 10 and _kana(t)),
    ("ja", "ひらがなだけで「ありがとう」と書いてください。漢字やカタカナは使わないでください。",
     lambda t: _exact(t, "ありがとう")),
    ("ja", "カタカナだけで3つの外来語を、読点（、）で区切って書いてください。",
     lambda t: _sep_only(t, "、", 3)
     and all(re.fullmatch(r"[ァ-ヴー]+", x.strip())
             for x in t.strip().rstrip("。").split("、"))),
    ("ja", "ちょうど10文字で夏を描写してください。句読点は使わないでください。",
     lambda t: _nchar(re.sub(r"[、。！？]", "", t)) == 10),
    ("ja", "「猫」という漢字を使わずに猫について一文書いてください。",
     lambda t: "猫" not in t and _nchar(t) > 10 and _kana(t)),
    ("ja", "1文だけ書いてください。「。」はちょうど1つだけ使うこと：富士山について。",
     lambda t: t.strip().count("。") == 1 and t.strip().endswith("。")),

    # ------------------------------------------------------------------- en
    ("en", "List exactly three colors, separated by commas, with no other text.",
     lambda t: _sep_only(t, ",", 3)),
    ("en", "Answer with the single word YES in all capital letters. Nothing else.",
     lambda t: _exact(t, "YES")),
    ("en", "Describe a rainy evening in exactly two sentences ending with periods.",
     lambda t: t.strip().count(".") == 2 and t.strip().endswith(".")),
    ("en", "Explain machine learning in under 30 words.",
     lambda t: 3 < len(t.split()) <= 30
     and re.search(r"learn|data|model|algorithm", t, re.I) is not None),
    ("en", "Write about tea without using the letter 'e' anywhere in your answer.",
     lambda t: "e" not in t.lower() and len(t.strip()) > 40),
    ("en", "Reply with valid JSON only: an object with keys \"city\" and \"country\" "
           "for Taipei. No markdown fences, no explanation.",
     lambda t: _is_json_with(t, {"city", "country"})),
    ("en", "What is 3 plus 4? Reply with a single digit and nothing else.",
     lambda t: _exact(t, "7")),
    ("en", "Name three countries, one per line, no numbering, no other text.",
     lambda t: len(_lines(t)) == 3 and all(len(x.split()) <= 3 for x in _lines(t))),
    ("en", "Write one sentence about the ocean using no punctuation at all.",
     lambda t: not re.search(r"[.,;:!?]", t) and len(t.split()) > 3),
    ("en", "Begin your answer with the exact words \"The answer is\": "
           "how many months are in a year?",
     lambda t: t.strip().startswith("The answer is") and "12" in t),
    ("en", "Write the numbers 1 through 5 separated by commas with no spaces "
           "and no other text.",
     lambda t: _exact(t.replace(" ", ""), "1,2,3,4,5")),
    ("en", "Write about spring without using any digits.",
     lambda t: not re.search(r"\d", t) and len(t.split()) > 15),
    ("en", "Write the word \"hello\" three times separated by commas, nothing else.",
     lambda t: _sep_only(t, ",", 3) and t.lower().count("hello") == 3),
    ("en", "Is the Earth larger than the Moon? Answer with one word only.",
     lambda t: _exact(t.lower(), "yes")),
    ("en", "Write a sentence that starts with \"Taiwan\" and ends with the word "
           "\"island\".",
     lambda t: t.strip().startswith("Taiwan")
     and t.strip().rstrip(".!").lower().endswith("island")),
    ("en", "Answer with a question (your reply must end with a question mark): "
           "which programming language should I learn?",
     lambda t: t.strip().endswith("?") and len(t.split()) > 2),
    ("en", "Answer in 20 words or fewer and include the word \"because\": "
           "why is the sky blue?",
     lambda t: "because" in t.lower() and len(t.split()) <= 20),
    ("en", "List four fruits, one per line, no numbering and no explanation.",
     lambda t: len(_lines(t)) == 4 and all(len(x.split()) <= 3 for x in _lines(t))),
    ("en", "List three sports as a bullet list where every line starts with \"- \".",
     lambda t: len(_lines(t)) == 3 and all(x.startswith("- ") for x in _lines(t))),
    ("en", "Reply with one word in all capital letters: what is the English name "
           "of the country whose capital is Tokyo?",
     lambda t: _exact(t, "JAPAN")),
    ("en", "Write two sentences about night markets. The first must start with "
           "\"Night markets\".",
     lambda t: t.strip().count(".") == 2 and t.strip().startswith("Night markets")),
    ("en", "Rewrite this as a question and output only the result: He came today.",
     lambda t: t.strip().endswith("?") and len(t.split()) <= 8),
    ("en", "Explain typhoons in between 30 and 60 words.",
     lambda t: 30 <= len(t.split()) <= 60),
    ("en", "Write five even numbers separated by commas, nothing else.",
     lambda t: _sep_only(t, ",", 5)
     and all(x.strip().isdigit() and int(x) % 2 == 0
             for x in t.strip().rstrip(".").split(","))),
    ("en", "Answer on a single line with no line breaks: what is cloud computing?",
     lambda t: "\n" not in t.strip() and len(t.split()) > 5),
    ("en", "Reply in all lowercase letters only: describe a cat in one sentence.",
     lambda t: t.strip() == t.strip().lower() and len(t.split()) > 4),
    ("en", "Write exactly ten words describing summer. No punctuation.",
     lambda t: len(re.sub(r"[.,;:!?]", "", t).split()) == 10),
    ("en", "Write one sentence about dogs without using the word \"dog\".",
     lambda t: "dog" not in t.lower() and len(t.split()) > 5),
    ("en", "Reply with exactly one sentence containing exactly one period: "
           "tell me about Mount Fuji.",
     lambda t: t.strip().count(".") == 1 and t.strip().endswith(".")
     and "fuji" in t.lower()),
    ("en", "Output only the three letters ABC in uppercase, nothing else.",
     lambda t: _exact(t, "ABC")),
]


def _is_json_with(text, keys):
    try:
        obj = json.loads(text.strip())
    except Exception:
        return False
    return isinstance(obj, dict) and keys <= set(obj)


# ------------------------------------------- 軸 C：通用能力保留 + 退化成翻譯機偵測

# 每語言 30 題（v4 那份 zhtw 5 / ja 3 / en 4，一題 8.3 個百分點，噪音大過訊號）。
# 答案唯一、可字串比對；數值題一律容許千分位與全形數字寫法。
GENERAL = [
    # (問題, 判分函數, 提示語言) — 答案唯一、可字串比對
    # ---------------------------------------------------------------- zh-TW
    ("台灣最高的山是哪一座？", lambda t: "玉山" in t, "zhtw"),
    ("一件衣服原價 1200 元，打七折後再折 100 元，最後要付多少元？只要數字。",
     lambda t: "740" in _num(t), "zhtw"),
    ("如果今天是星期三，三天後是星期幾？", lambda t: "六" in t, "zhtw"),
    ("台灣的首都是哪裡？", lambda t: "台北" in t or "臺北" in t, "zhtw"),
    ("水在攝氏幾度會沸騰？（標準大氣壓）", lambda t: "100" in _num(t), "zhtw"),
    ("一打是幾個？", lambda t: "12" in _num(t) or "十二" in t, "zhtw"),
    ("台灣最長的河流是哪一條？", lambda t: "濁水溪" in t, "zhtw"),
    ("光合作用主要發生在植物的哪個部位？", lambda t: "葉" in t, "zhtw"),
    ("3 的 4 次方是多少？", lambda t: "81" in _num(t), "zhtw"),
    ("一年有幾天？（平年）", lambda t: "365" in _num(t), "zhtw"),
    ("《紅樓夢》的作者是誰？", lambda t: "曹雪芹" in t, "zhtw"),
    ("水的化學式是什麼？", lambda t: "H2O" in t.upper().replace("₂", "2"), "zhtw"),
    ("台灣的貨幣單位是什麼？",
     lambda t: "新台幣" in t or "新臺幣" in t or "台幣" in t or "臺幣" in t, "zhtw"),
    ("如果一個正方形的邊長是 5 公分，面積是多少平方公分？只要數字。",
     lambda t: "25" in _num(t), "zhtw"),
    ("人體正常體溫大約是攝氏幾度？", lambda t: "36" in _num(t) or "37" in _num(t), "zhtw"),
    ("太陽系裡離太陽最近的行星是哪一顆？", lambda t: "水星" in t, "zhtw"),
    ("中華民國是哪一年建立的？（西元）", lambda t: "1912" in _num(t), "zhtw"),
    ("一公斤等於幾公克？", lambda t: "1000" in _num(t) or "一千" in t, "zhtw"),
    ("珠穆朗瑪峰（聖母峰）位於哪兩個國家的邊界？",
     lambda t: ("尼泊爾" in t) and ("中國" in t or "西藏" in t), "zhtw"),
    ("如果 x + 7 = 15，x 是多少？只要數字。", lambda t: "8" in _num(t), "zhtw"),
    ("寫一個 Python 函式，回傳一個串列裡所有偶數的總和。",
     lambda t: "def" in t and "%" in t and "2" in t, "zhtw"),
    ("colour 的美式英語拼法是什麼？", lambda t: "color" in t.lower(), "zhtw"),
    ("哪一種血型被稱為萬能捐血者？", lambda t: "O" in t and ("陰" in t or "O型" in t
                                                or "O 型" in t), "zhtw"),
    ("台灣位於哪一個大洋的西側？", lambda t: "太平洋" in t, "zhtw"),
    ("一星期有幾個小時？只要數字。", lambda t: "168" in _num(t), "zhtw"),
    ("請把「今天天氣很好」這句話改成否定句。",
     lambda t: "不" in t or "沒" in t, "zhtw"),
    ("鑽石和石墨都是由哪一種元素構成的？", lambda t: "碳" in t, "zhtw"),
    ("如果一本書有 240 頁，我讀了四分之三，還剩幾頁？只要數字。",
     lambda t: "60" in _num(t), "zhtw"),
    ("農曆新年又叫做什麼節？", lambda t: "春節" in t, "zhtw"),
    ("光在真空中的速度大約是每秒幾公里？",
     lambda t: "300000" in _num(t) or "30萬" in t or "三十萬" in t, "zhtw"),

    # ------------------------------------------------------------------- ja
    ("日本の首都はどこですか。", lambda t: "東京" in t, "ja"),
    ("1年は何ヶ月ですか。", lambda t: "12" in _num(t) or "十二" in t, "ja"),
    ("富士山の高さは約何メートルですか。", lambda t: "3776" in _num(t), "ja"),
    ("日本で一番長い川は何ですか。", lambda t: "信濃" in t, "ja"),
    ("水は摂氏何度で沸騰しますか。（1気圧）", lambda t: "100" in _num(t), "ja"),
    ("3の4乗はいくつですか。", lambda t: "81" in _num(t), "ja"),
    ("1年は何日ですか。（平年）", lambda t: "365" in _num(t), "ja"),
    ("water の化学式は何ですか。",
     lambda t: "H2O" in t.upper().replace("₂", "2"), "ja"),
    ("日本の通貨単位は何ですか。", lambda t: "円" in t or "エン" in t, "ja"),
    ("一辺が5センチの正方形の面積は何平方センチですか。数字だけ。",
     lambda t: "25" in _num(t), "ja"),
    ("太陽に一番近い惑星は何ですか。", lambda t: "水星" in t, "ja"),
    ("1キログラムは何グラムですか。", lambda t: "1000" in _num(t) or "千" in t, "ja"),
    ("x + 7 = 15 のとき、x はいくつですか。数字だけ。", lambda t: "8" in _num(t), "ja"),
    ("リスト内の偶数の合計を返すPython関数を書いてください。",
     lambda t: "def" in t and "%" in t and "2" in t, "ja"),
    ("『源氏物語』の作者は誰ですか。", lambda t: "紫式部" in t, "ja"),
    ("ダイヤモンドと黒鉛は何という元素からできていますか。", lambda t: "炭素" in t, "ja"),
    ("1週間は何時間ですか。数字だけ。", lambda t: "168" in _num(t), "ja"),
    ("日本の最も北にある都道府県はどこですか。", lambda t: "北海道" in t, "ja"),
    ("光合成は植物のどの部分で主に行われますか。", lambda t: "葉" in t, "ja"),
    ("240ページの本の4分の3を読みました。残りは何ページですか。数字だけ。",
     lambda t: "60" in _num(t), "ja"),
    ("明治時代は西暦何年に始まりましたか。", lambda t: "1868" in _num(t), "ja"),
    ("人間の平熱はおよそ摂氏何度ですか。",
     lambda t: "36" in _num(t) or "37" in _num(t), "ja"),
    ("「今日は天気がいい」を否定文にしてください。",
     lambda t: "ない" in t or "よくない" in t or "悪い" in t, "ja"),
    ("エベレストはどの2つの国の国境にありますか。",
     lambda t: "ネパール" in t and ("中国" in t or "チベット" in t), "ja"),
    ("日本の国花とされる花を1つ挙げてください。",
     lambda t: "桜" in t or "サクラ" in t or "菊" in t, "ja"),
    ("真空中の光の速さはおよそ秒速何キロメートルですか。",
     lambda t: "300000" in _num(t) or "30万" in t or "三十万" in t, "ja"),
    ("17 かける 3 はいくつですか。", lambda t: "51" in _num(t), "ja"),
    ("太陽系で一番大きい惑星は何ですか。", lambda t: "木星" in t, "ja"),
    ("日本語の「ありがとう」は英語で何と言いますか。",
     lambda t: "thank" in t.lower(), "ja"),
    ("酸素の元素記号は何ですか。",
     lambda t: re.search(r"\bO\b", t) is not None, "ja"),

    # ------------------------------------------------------------------- en
    ("What is the capital of Japan?", lambda t: "Tokyo" in t, "en"),
    ("What is 17 multiplied by 3?", lambda t: "51" in _num(t), "en"),
    ("Write a Python function that returns the sum of even numbers in a list.",
     lambda t: "def" in t and "%" in t and "2" in t, "en"),
    ("Name the largest planet in our solar system.", lambda t: "Jupiter" in t, "en"),
    ("At what temperature in Celsius does water boil at sea level?",
     lambda t: "100" in _num(t), "en"),
    ("What is 3 to the power of 4?", lambda t: "81" in _num(t), "en"),
    ("How many days are in a common year?", lambda t: "365" in _num(t), "en"),
    ("What is the chemical formula for water?",
     lambda t: "H2O" in t.upper().replace("₂", "2"), "en"),
    ("A square has sides of 5 cm. What is its area in square centimetres? "
     "Number only.", lambda t: "25" in _num(t), "en"),
    ("Which planet is closest to the Sun?", lambda t: "Mercury" in t, "en"),
    ("How many grams are in a kilogram?", lambda t: "1000" in _num(t), "en"),
    ("If x + 7 = 15, what is x? Number only.", lambda t: "8" in _num(t), "en"),
    ("Who wrote the play Romeo and Juliet?", lambda t: "Shakespeare" in t, "en"),
    ("What element are both diamond and graphite made of?",
     lambda t: "carbon" in t.lower(), "en"),
    ("How many hours are in a week? Number only.", lambda t: "168" in _num(t), "en"),
    ("In which part of a plant does photosynthesis mainly occur?",
     lambda t: "leaf" in t.lower() or "leaves" in t.lower(), "en"),
    ("A book has 240 pages and I have read three quarters of it. "
     "How many pages are left? Number only.", lambda t: "60" in _num(t), "en"),
    ("In what year did the Second World War end?", lambda t: "1945" in _num(t), "en"),
    ("What is normal human body temperature in degrees Celsius?",
     lambda t: "36" in _num(t) or "37" in _num(t), "en"),
    ("Make this sentence negative: The weather is nice today.",
     lambda t: "weather" in t.lower()
     and re.search(r"\bnot\b|n't", t, re.I) is not None, "en"),
    ("Everest sits on the border of which two countries?",
     lambda t: "Nepal" in t and ("China" in t or "Tibet" in t), "en"),
    ("Roughly how fast does light travel in a vacuum, in kilometres per second?",
     lambda t: "300000" in _num(t) or "299792" in _num(t), "en"),
    ("What is the chemical symbol for oxygen?",
     lambda t: re.search(r"\bO\b", t) is not None, "en"),
    ("What is the longest river in Japan?", lambda t: "Shinano" in t, "en"),
    ("Convert 25 degrees Celsius to Fahrenheit. Number only.",
     lambda t: "77" in _num(t), "en"),
    ("How many sides does a hexagon have?",
     lambda t: "6" in _num(t) or "six" in t.lower(), "en"),
    ("Which gas do plants absorb from the air for photosynthesis?",
     lambda t: "carbon dioxide" in t.lower() or "CO2" in t.upper().replace("₂", "2"),
     "en"),
    ("What does HTTP stand for?",
     lambda t: "hypertext" in t.lower().replace(" ", "").replace("-", "")
     or "hyper text" in t.lower(), "en"),
    ("Sort these numbers in ascending order: 5, 2, 9, 1. Numbers only.",
     lambda t: re.sub(r"\D", "", _num(t)).startswith("1259"), "en"),
    ("What is the largest ocean on Earth?", lambda t: "Pacific" in t, "en"),
]


def is_mere_translation(prompt, answer):
    """偵測「模型把問題翻譯掉而不是回答」——v2/v3 的招牌失敗模式。

    判據：答案長度與問題相近（0.5~2.0 倍）、答案主要文字系統與問題不同、
    且答案結尾仍是問句。任一組合成立即視為翻譯機行為。"""
    p, a = prompt.strip(), answer.strip()
    ratio = len(a) / max(len(p), 1)
    if not (0.4 <= ratio <= 2.5):
        return False
    han = re.compile(r"[一-鿿]")
    kana = re.compile(r"[぀-ヿ]")
    latin = re.compile(r"[A-Za-z]")
    # 純數字/算式答案不算翻譯（答錯是答錯，不是翻譯機）
    if len(han.findall(a)) + len(kana.findall(a)) + len(latin.findall(a)) < 5:
        return False

    def sig(s):
        return (bool(kana.search(s)), bool(han.search(s)),
                len(latin.findall(s)) > len(s) * 0.3)
    switched = sig(p) != sig(a)
    still_question = a.rstrip().endswith(("？", "?"))
    return switched or still_question


# ------------------------------------------------------------------------ 推論

def stop_token_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if tid is not None and tid != tok.unk_token_id:
            ids.add(tid)
    return sorted(ids)


GEN_KW = {}  # CLI 覆寫的解碼參數（見 main），優先於 DECODE 的每目標語言預設


def generate(tok, model, convs, max_new, batch, gen=None):
    """gen：該方向的解碼預設（evaluate.DECODE[目標語言]）。ifeval/general 兩軸
    不是翻譯、沒有單一目標語言，傳 None 走 greedy。"""
    eos = stop_token_ids(tok)
    kw = {**(gen or {}), **GEN_KW}
    out = []
    for i in range(0, len(convs), batch):
        inp = tok.apply_chat_template(convs[i:i + batch], add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt",
                                      padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                               eos_token_id=eos, pad_token_id=tok.pad_token_id,
                               **kw)
        n = inp["input_ids"].shape[1]
        out.extend(tok.decode(x[n:], skip_special_tokens=True).strip() for x in g)
        print(f"    {min(i + batch, len(convs))}/{len(convs)}", flush=True)
    return out


def run_doc(tok, model, n_docs, batch, hyp_dir):
    print("== 軸 A：文件級翻譯（WMT24++）==")
    pairs = load_docs(n_docs)
    results = {}
    for (src_l, tgt_l), rows in pairs.items():
        name = f"{src_l}->{tgt_l}"
        print(f"  [{name}] {len(rows)} 篇")
        srcs = [s for s, _ in rows]
        refs = [r for _, r in rows]
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"{INSTR[tgt_l]}\n{s}"}] for s in srcs]
        hyps = generate(tok, model, convs, max_new=2048, batch=batch,
                        gen=DECODE[tgt_l])
        results[name] = score_doc((src_l, tgt_l), srcs, hyps, refs)
        print(f"    {results[name]}")
        stem = name.replace("->", "2")
        for suf, rows_ in (("src", srcs), ("ref", refs), ("hyp", hyps)):
            (hyp_dir / f"{stem}.{suf}.txt").write_text(
                "\n<<<DOC>>>\n".join(rows_), encoding="utf-8", newline="\n")
    return results


def run_ifeval(tok, model, batch, hyp_dir):
    print("== 軸 B：可驗證指令跟隨 ==")
    convs = [[{"role": "user", "content": q}] for _, q, _ in IFEVAL]
    outs = generate(tok, model, convs, max_new=512, batch=batch)
    per_lang, detail = Counter(), []
    tot = Counter()
    for (lang, q, check), o in zip(IFEVAL, outs):
        try:
            ok = bool(check(o))
        except Exception:
            ok = False
        per_lang[lang] += ok
        tot[lang] += 1
        detail.append({"lang": lang, "instruction": q, "output": o, "pass": ok})
    (hyp_dir / "ifeval.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    res = {lg: round(per_lang[lg] / tot[lg] * 100, 1) for lg in tot}
    res["overall"] = round(sum(per_lang.values()) / sum(tot.values()) * 100, 1)
    print(f"  {res}")
    return res


def run_general(tok, model, batch, hyp_dir):
    print("== 軸 C：通用能力保留 ==")
    convs = [[{"role": "user", "content": q}] for q, _, _ in GENERAL]
    outs = generate(tok, model, convs, max_new=512, batch=batch)
    correct, mere = 0, 0
    per_lang, tot = Counter(), Counter()
    detail = []
    for (q, check, lang), o in zip(GENERAL, outs):
        try:
            ok = bool(check(o))
        except Exception:
            ok = False
        mt = is_mere_translation(q, o)
        correct += ok
        mere += mt
        per_lang[lang] += ok
        tot[lang] += 1
        detail.append({"lang": lang, "question": q, "output": o,
                       "correct": ok, "mere_translation": mt})
    (hyp_dir / "general.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    n = len(GENERAL)
    # 分語言看：災難性遺忘在 v3 是中文側先垮，總分會把它平均掉
    res = {"accuracy_pct": round(correct / n * 100, 1),
           **{f"acc_{lg}_pct": round(per_lang[lg] / tot[lg] * 100, 1) for lg in tot},
           "mere_translation_pct": round(mere / n * 100, 1), "n": n}
    print(f"  {res}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--axis", default="all", choices=["all", "doc", "ifeval", "general"])
    ap.add_argument("--docs", type=int, default=25, help="文件級評測取幾篇")
    ap.add_argument("--batch", type=int, default=4)
    # v4 的軸 A 失效模式是「少數文件陷入 greedy 迴圈整篇報廢」（en->zhtw 4/12 篇），
    # 這兩個旗標就是用來驗證那是解碼問題還是翻譯能力問題。用不同 --tag 跑，別覆蓋原始結果。
    ap.add_argument("--rep-penalty", type=float, default=None)
    ap.add_argument("--no-repeat-ngram", type=int, default=None)
    args = ap.parse_args()

    if args.rep_penalty:
        GEN_KW["repetition_penalty"] = args.rep_penalty
    if args.no_repeat_ngram:
        GEN_KW["no_repeat_ngram_size"] = args.no_repeat_ngram

    print("== loading model ==")
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  adapter: {args.adapter}")
    model.eval()

    hyp_dir = ROOT / "results" / "capability" / args.tag
    hyp_dir.mkdir(parents=True, exist_ok=True)
    dest = ROOT / "results" / "capability" / f"{args.tag}.json"
    # 單軸執行時保留其他軸既有結果，不覆蓋
    out = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {}
    # 解碼參數一定要跟著結果存：同一顆 adapter 換 rep-penalty 分數會差很多，
    # 沒記下來的話事後分不出「哪一組數字是哪種解碼跑的」
    out |= {"tag": args.tag, "model": args.model, "adapter": args.adapter,
            "gen": dict(GEN_KW), "decode_defaults": DECODE}
    axes = ["doc", "ifeval", "general"] if args.axis == "all" else [args.axis]
    if "doc" in axes:
        out["doc"] = run_doc(tok, model, args.docs, args.batch, hyp_dir)
    if "ifeval" in axes:
        out["ifeval"] = run_ifeval(tok, model, args.batch, hyp_dir)
    if "general" in axes:
        out["general"] = run_general(tok, model, args.batch, hyp_dir)

    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults -> {dest}")


if __name__ == "__main__":
    main()
