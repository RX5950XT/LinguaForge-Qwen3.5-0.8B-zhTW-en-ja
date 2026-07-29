"""evaluate.py 的簡體洩漏偵測自檢。

舊版用 s2tw 整句 round-trip，把台灣正字（了、布、岩、污、周…）判成洩漏；
這裡把當時實際誤判的句子與真正含簡體的句子都釘住，避免改回去。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import _SIMPLIFIED, TW_VARIANTS  # noqa: E402


def leaks(s: str) -> bool:
    return any(c in _SIMPLIFIED for c in s)


CLEAN = [  # v5c 實際輸出中被舊規則誤判的句子
    "他說他發明了一個用Wi-Fi發聲的門鈴。",
    "在1960年代，布雷希尼斯基擔任約翰·F·肯尼迪的顧問。",
    "在旅途中，岩崎先生也多次遇到困難。",
    "胡錦濤國家主席呼籲發展中國家「要從治污做起，避免重蹈覆轍」。",
    "社區管理員Adam Carden在上周訪問維基新媒體時表示。",
    "一群人在裡面吃飯，游泳池旁的床邊有個托盤。",
    "台北到高雄約三百公里，朴槿惠曾來訪。",
]
DIRTY = [  # 真簡體，必須抓到
    "2010年大選中，當女王伊丽莎白二世駕崩。",
    "以昆蟲、啮齒動物、蜥蜴、鳥類等小型獵物為食。",
    "这个国家的发展很快。",
    "馬爾克斯的书很有名。",
]


def main():
    for s in CLEAN:
        assert not leaks(s), f"誤判為洩漏: {s}  命中 {[c for c in s if c in _SIMPLIFIED]}"
    for s in DIRTY:
        assert leaks(s), f"漏抓簡體: {s}"
    # 白名單裡的字必須真的是「s2t 會改、t2s 不改」那類，否則放在名單裡沒意義
    from evaluate import cc_s2t, cc_t2s
    for c in TW_VARIANTS:
        assert cc_t2s.convert(c) == c and cc_s2t.convert(c) != c, f"{c} 不需列入白名單"
    print(f"OK: {len(CLEAN)} 句正字未誤判、{len(DIRTY)} 句簡體全抓到、"
          f"白名單 {len(TW_VARIANTS)} 字皆有效（簡體字表 {len(_SIMPLIFIED):,} 字）")


if __name__ == "__main__":
    main()
