"""stage2_zeroshot.count_hits 도메인-불가지 헬퍼 테스트."""
from __future__ import annotations

import unittest

from src.classify.medical_terms import MEDICAL_TERMS
from src.classify.stage2_zeroshot import count_hits


class TestCountHits(unittest.TestCase):
    def test_korean_substring_hits(self) -> None:
        n, terms = count_hits("환자 진단 처방 내역", MEDICAL_TERMS)
        self.assertGreaterEqual(n, 2)
        self.assertIn("환자", terms)

    def test_latin_word_boundary(self) -> None:
        n, terms = count_hits("CT scan and MRI ordered", MEDICAL_TERMS)
        self.assertIn("ct", terms)
        self.assertIn("mri", terms)

    def test_latin_substring_not_matched(self) -> None:
        # 'ct' 가 'contact'/'factory' 안에 있어도 매칭 안 됨(단어 경계).
        n, _ = count_hits("Please contact the factory team", MEDICAL_TERMS)
        self.assertEqual(n, 0)

    def test_zero_hits(self) -> None:
        n, terms = count_hits("주식 시장 동향 분석", MEDICAL_TERMS)
        self.assertEqual(n, 0)
        self.assertEqual(terms, [])

    def test_arbitrary_lexicon(self) -> None:
        n, terms = count_hits("이번 계약 소송 건", frozenset({"계약", "소송"}))
        self.assertEqual(n, 2)

    def test_empty_lexicon(self) -> None:
        n, terms = count_hits("anything", frozenset())
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
