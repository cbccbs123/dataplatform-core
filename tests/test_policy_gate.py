"""A2 — policy_gate 오탐 제거: 문자열/주석/f-string 리터럴은 검사에서 빼되 실 코드 위반은 계속 잡는다.

기존 게이트는 `#` 주석만 제거하고 문자열/독스트링은 그대로 스캔해, 정책 설명 문구('fine-tuning 배제')나
`ImageOps.fit(` 언급까지 학습 흔적으로 오탐했다. `tokenize` 로 STRING/COMMENT/FSTRING 토큰을 마스킹한
`_code_only_lines` 를 검증한다 — 리터럴은 가려지고, 실제 코드 토큰(loss.backward·실제 .fit( 호출)은 남아야 한다.
"""

from __future__ import annotations

import re
import unittest

import scripts.policy_gate as pg


def _line1(src: str) -> str:
    """소스의 1행에 대한 '코드만' 마스킹 결과."""
    return pg._code_only_lines(src)[1]


class TestCodeOnlyMasking(unittest.TestCase):
    def test_string_literal_finetune_masked(self) -> None:
        self.assertNotIn("fine-tune", _line1('x = "we never fine-tune"\n'))

    def test_docstring_finetune_masked(self) -> None:
        self.assertNotIn("fine", _line1('"""정책: fine-tuning 배제."""\n'))

    def test_comment_masked(self) -> None:
        self.assertNotIn("optimizer.step", _line1("z = 1  # optimizer.step 언급\n"))

    def test_real_code_violation_preserved(self) -> None:
        # 실제 코드 토큰은 남아야 검출된다(게이트 무력화 금지).
        self.assertIn("loss.backward", _line1("loss.backward()\n"))

    def test_fit_in_string_masked_but_real_call_preserved(self) -> None:
        self.assertNotIn(".fit(", _line1('doc = "ImageOps.fit(x)"\n'))  # 문자열 속 언급은 무시
        self.assertIn(".fit(", _line1("clf.fit(X)\n"))  # 실제 호출은 보존

    def test_fstring_literal_masked_but_expr_preserved(self) -> None:
        code = _line1('v = f"note fine-tune {loss.backward}"\n')
        self.assertNotIn("fine-tune", code)  # f-string 리터럴부 마스킹
        self.assertIn("loss.backward", code)  # f-string 내 표현식(코드)은 보존


class TestGateBehaviorWithMasking(unittest.TestCase):
    """마스킹이 실제 게이트 판정에 반영되는지 — CHECKS 정규식으로 확인."""

    _RX = pg.CHECKS[0][1]  # 학습 배제 정규식

    def test_finetune_in_docstring_not_matched(self) -> None:
        self.assertFalse(self._RX.search(_line1('"""fine-tuning 은 배제한다."""\n')))

    def test_finetune_in_real_code_matched(self) -> None:
        # 코드 식별자에 fine_tune 이 있으면 검출 대상(마스킹 안 됨)
        self.assertTrue(self._RX.search(_line1("do_fine_tune()\n")))

    def test_temperature_in_string_not_flagged(self) -> None:
        self.assertFalse(pg.TEMP_RX.search(_line1('msg = "temperature=0.7 예시"\n')))

    def test_temperature_in_code_flagged(self) -> None:
        m = pg.TEMP_RX.search(_line1("call(temperature=0.7)\n"))
        self.assertTrue(m and float(m.group(1)) != 0)


if __name__ == "__main__":
    unittest.main()
