"""eval_capability 判分函數自檢（不載模型、不吃 GPU）。

擴到每語言 30 題後，判分邏輯本身就是評測的可信度來源。兩個方向都要擋：
  1. 太寬鬆——空字串或英文拒答不得通過任何題目（否定型約束最容易 vacuously pass）
  2. 太嚴格——人工寫的正確答案必須判成通過，否則模型再好也拿不到分
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from eval_capability import GENERAL, IFEVAL, alignment  # noqa: E402

# --- alignment 必須把「尾段腰斬」跟「多吐垃圾行」分開 -------------------------
# 這正是 base-doc25 騙過 completeness_median 的手法：整篇字元比 ~1.0，
# 實際是前段超譯 + 後段腰斬 + 多吐 40% 的行互相抵消。
_REF = ["\n".join(f"reference line {i} with some length" for i in range(9))]
_TRUNC = ["\n".join(f"translated line {i} with some length" if i < 6 else ""
                    for i in range(9))]                       # 後 1/3 空白
_JUNK = [_REF[0] + "\n" + "\n".join(f"note {i}" for i in range(5))]   # 多吐 5 行

lr, tr = alignment(_TRUNC, _REF)
assert lr == 1.0, f"行數沒少，line_ratio 應為 1.0：{lr}"
assert tr == 0.0, f"後 1/3 全空，tail_ratio 應為 0.0：{tr}"

lr, tr = alignment(_JUNK, _REF)
assert lr > 1.5, f"多吐 5 行沒被 line_ratio 抓到：{lr}"
assert tr >= 1.0, f"垃圾行接在尾巴，tail_ratio 應 >= 1.0：{tr}"

lr, tr = alignment(_REF, _REF)
assert (lr, tr) == (1.0, 1.0), f"完全一致應為 (1.0, 1.0)：{lr}, {tr}"

# --- 面板規模與唯一性 --------------------------------------------------------
assert len(IFEVAL) == 90, len(IFEVAL)
assert len(GENERAL) == 90, len(GENERAL)
assert Counter(l for l, _, _ in IFEVAL) == {"zhtw": 30, "ja": 30, "en": 30}
assert Counter(l for _, _, l in GENERAL) == {"zhtw": 30, "ja": 30, "en": 30}
assert len({q for _, q, _ in IFEVAL}) == 90, "ifeval 有重複題目"
assert len({q for q, _, _ in GENERAL}) == 90, "general 有重複題目"


def _check(fn, text):
    try:
        return bool(fn(text))
    except Exception:
        return False


# --- 1. 沒有一題可以被廢話通過 ------------------------------------------------
JUNK = ["", "   ", "\n\n", "I cannot answer that.", "Sorry, I do not know.",
        "...", "無法回答", "すみません"]
vacuous = [q for junk in JUNK
           for lang, q, fn in IFEVAL if _check(fn, junk)]
vacuous += [q for junk in JUNK
            for q, fn, lang in GENERAL if _check(fn, junk)]
assert not vacuous, f"{len(vacuous)} 題可被廢話通過: {vacuous[:3]}"

# --- 2. 人工正確答案必須通過（挑各種判分形狀的代表題）-------------------------
IF = {q: fn for _, q, fn in IFEVAL}
GE = {q: fn for q, fn, _ in GENERAL}

POSITIVE = [
    # 分隔符與數量
    (IF["列出三種台灣常見的水果，只要名稱，用頓號分隔，不要任何其他文字。"], "芒果、香蕉、鳳梨"),
    (IF["列出三個顏色，用全形逗號「，」分隔，不要頓號，不要其他文字。"], "紅，藍，綠"),
    (IF["把「你好」重複三次，用頓號分隔，不要其他文字。"], "你好、你好、你好"),
    (IF["列出五個數字，用頓號分隔，每個都必須是偶數，不要其他文字。"], "2、4、6、8、10"),
    (IF["List exactly three colors, separated by commas, with no other text."],
     "red, blue, green"),
    (IF["Write five even numbers separated by commas, nothing else."],
     "2, 4, 6, 8, 10"),
    (IF["カタカナだけで3つの外来語を、読点（、）で区切って書いてください。"],
     "パン、コーヒー、テレビ"),
    # 完全相符
    (IF["回答「台北」這兩個字，不要標點，不要其他任何文字。"], "臺北"),
    (IF["3 加 4 等於多少？只回答一個阿拉伯數字，不要其他文字。"], "7"),
    (IF["ひらがなだけで「ありがとう」と書いてください。漢字やカタカナは使わないでください。"],
     "ありがとう"),
    (IF["Output only the three letters ABC in uppercase, nothing else."], "ABC"),
    # 行數與前綴
    (IF["列出三個台灣的縣市，每行一個，不要編號，不要其他文字。"], "台北市\n新竹縣\n花蓮縣"),
    (IF["用條列式列出三項運動，每行都要以「- 」開頭。"], "- 游泳\n- 跑步\n- 籃球"),
    (IF["寫出四個四字成語，每行一個，不要編號，不要解釋。"],
     "一石二鳥\n畫蛇添足\n守株待兔\n亡羊補牢"),
    (IF["List four fruits, one per line, no numbering and no explanation."],
     "apple\nbanana\ncherry\ngrape"),
    # 長度
    (IF["用剛好十個字描述夏天，不要標點符號。"], "陽光灑落海面波光粼粼"),
    (IF["Write exactly ten words describing summer. No punctuation."],
     "warm bright days long evenings loud cicadas endless blue skies"),
    (IF["用繁體中文回答：什麼是機器學習？回答不超過 50 個字。"],
     "讓電腦從資料中自行找出規律，不必逐條寫死規則。"),
    (IF["Explain machine learning in under 30 words."],
     "It is a way for computers to learn patterns from data instead of "
     "following rules written by hand."),
    # 否定約束
    (IF["寫一段關於咖啡的介紹，全文不可以出現「咖啡」這兩個字。"],
     "這種黑色飲料由烘焙過的豆子沖泡而成，帶有明顯的苦味與香氣，是許多人早晨的習慣。"),
    (IF["Write about tea without using the letter 'e' anywhere in your answer."],
     "A hot drink from dry plant bits, put in a cup, with a soft calming aroma."),
    (IF["「猫」という漢字を使わずに猫について一文書いてください。"],
     "ネコはとても静かな動物で、家の中でよく眠っています。"),
    # 起訖與句數
    (IF["寫一句話，必須以「台灣」開頭、以「島」字結尾。"], "台灣是太平洋西側的一座島"),
    (IF["Write a sentence that starts with \"Taiwan\" and ends with the word "
        "\"island\"."], "Taiwan is a mountainous island"),
    (IF["用剛好兩句話描述下雨的傍晚，每句都要以「。」結尾。"], "雨絲落在窗上。街燈慢慢亮起。"),
    (IF["Reply with exactly one sentence containing exactly one period: "
        "tell me about Mount Fuji."], "Mount Fuji is the tallest peak in Japan."),
    # 格式
    (IF["只回答 JSON，不要 markdown 圍欄也不要說明：一個物件，"
        "包含 \"city\" 與 \"country\" 兩個鍵，內容是台北。"],
     '{"city": "Taipei", "country": "Taiwan"}'),
    (IF["回答時每個字之間都要空一個半形空格：你好嗎"], "你 好 嗎"),
    (IF["Reply in all lowercase letters only: describe a cat in one sentence."],
     "a small quiet animal that sleeps most of the day"),
    # GENERAL：數值正規化（千分位、全形）不可判錯
    (GE["一公斤等於幾公克？"], "1,000 公克"),
    (GE["1キログラムは何グラムですか。"], "１０００グラムです。"),
    (GE["How many grams are in a kilogram?"], "There are 1,000 grams."),
    (GE["Sort these numbers in ascending order: 5, 2, 9, 1. Numbers only."],
     "1, 2, 5, 9"),
    (GE["水的化學式是什麼？"], "水的化學式是 H₂O。"),
    (GE["Make this sentence negative: The weather is nice today."],
     "The weather is not nice today."),
    (GE["「今日は天気がいい」を否定文にしてください。"], "今日は天気がよくない。"),
    (GE["酸素の元素記号は何ですか。"], "酸素の元素記号は O です。"),
]
for i, (fn, ans) in enumerate(POSITIVE):
    assert _check(fn, ans), f"POSITIVE[{i}] 應該通過卻被判錯: {ans!r}"

# --- 3. 明確違規必須被擋下 ----------------------------------------------------
NEGATIVE = [
    (IF["列出三種台灣常見的水果，只要名稱，用頓號分隔，不要任何其他文字。"],
     "當然可以！以下是三種水果：芒果、香蕉、鳳梨"),          # 多了說明文字
    (IF["回答「台北」這兩個字，不要標點，不要其他任何文字。"], "答案是台北。"),
    (IF["用條列式列出三項運動，每行都要以「- 」開頭。"], "游泳\n跑步\n籃球"),
    (IF["寫一段關於咖啡的介紹，全文不可以出現「咖啡」這兩個字。"],
     "咖啡是一種由烘焙咖啡豆沖泡而成的飲料，風味濃郁，全世界都有人喝。"),
    (IF["Write exactly ten words describing summer. No punctuation."],
     "warm bright days"),
    (IF["Reply in all lowercase letters only: describe a cat in one sentence."],
     "A small quiet animal that sleeps most of the day"),
    (GE["台灣最高的山是哪一座？"], "台灣最高的山是阿里山。"),
    (GE["What is 17 multiplied by 3?"], "17 times 3 is 54."),
]
for i, (fn, ans) in enumerate(NEGATIVE):
    assert not _check(fn, ans), f"NEGATIVE[{i}] 應該被擋卻通過了: {ans!r}"

print(f"eval_capability checkers OK — ifeval {len(IFEVAL)} / general {len(GENERAL)}，"
      f"正例 {len(POSITIVE)}、反例 {len(NEGATIVE)}")
