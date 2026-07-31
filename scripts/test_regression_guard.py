"""regression_guard 判定邏輯自檢（不需 GPU、不讀 results/）。

夾兩邊：真退步必須擋下來，雜訊帶內必須判 TIE? 而不是靜靜放行或誤殺。

  uv run python scripts/test_regression_guard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from regression_guard import COMET_TOL, verdict  # noqa: E402

# COMET（min + 容忍帶）：贏、平手、真退步三段都要分得開
assert verdict(86.40, 86.32, "min", COMET_TOL) == "PASS"
assert verdict(86.32, 86.32, "min", COMET_TOL) == "PASS"          # 等於 base 算過
assert verdict(86.23, 86.32, "min", COMET_TOL) == "TIE?"          # v5e 的 en->zhtw
assert verdict(85.81, 86.32, "min", COMET_TOL) == "FAIL"          # 超出雜訊帶

# 洩漏（max，無容忍帶）：+0.3 界線上要過，超過要擋
assert verdict(1.09, 10.18 + 0.3, "max") == "PASS"
assert verdict(10.48, 10.18 + 0.3, "max") == "PASS"
assert verdict(10.49, 10.18 + 0.3, "max") == "FAIL"

# BELEBELE（min，無容忍帶）：v5f 的 ja 必須擋下來，v5e 的必須放行
assert verdict(43.94, 51.78 - 3.0, "min") == "FAIL"
assert verdict(52.36, 51.78 - 3.0, "min") == "PASS"
assert verdict(51.28, 55.81 - 3.0, "min") == "FAIL"               # v5f zh-TW

# doc 絕對閘：v3 的腰斬要擋，v5e 的要過
assert verdict(0.0, 0.80, "min") == "FAIL"                        # v3 尾段趨近 0
assert verdict(0.851, 0.80, "min") == "PASS"                      # v5e 最低的一格
assert verdict(91.7, 5.0, "max") == "FAIL"                        # v3 腰斬率
assert verdict(4.0, 5.0, "max") == "PASS"

# 無容忍帶時，差一點點就是 FAIL，不得偷偷判 TIE?
assert verdict(86.31, 86.32, "min") == "FAIL"

print("regression_guard 判定邏輯 OK — COMET 三段、洩漏、BELEBELE、doc 絕對閘皆符合預期")
