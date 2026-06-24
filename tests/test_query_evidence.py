"""044 G0/G2 — query_evidence 순수 함수 단위 테스트."""

from __future__ import annotations

import unittest

from src.search.query_evidence import (
    EVIDENCE_WEIGHTS,
    evidence_score,
    lexical_rescue_keep,
    strong_evidence_score,
)
from src.search.query_plan import build_search_policy


class EvidenceScoreTest(unittest.TestCase):
    def test_empty_returns_zero(self) -> None:
        self.assertEqual(evidence_score([]), 0.0)
        self.assertEqual(evidence_score(None), 0.0)

    def test_unknown_name_ignored(self) -> None:
        self.assertEqual(evidence_score(["hit_unknown", "not_a_field"]), 0.0)

    def test_weak_only_table(self) -> None:
        # 설계 §11.2 예: summary + search_text = 0.7 + 0.3
        self.assertAlmostEqual(
            evidence_score(["hit_summary", "hit_search_text"]),
            1.0,
        )

    def test_keywords_only(self) -> None:
        self.assertAlmostEqual(evidence_score(["hit_keywords"]), 3.0)

    def test_dedup_search_text_when_strong(self) -> None:
        # keywords hit 시 search_text 중복 가산 금지
        score = evidence_score(["hit_keywords", "hit_search_text"])
        self.assertAlmostEqual(score, 3.0)

    def test_strong_evidence_score(self) -> None:
        self.assertAlmostEqual(
            strong_evidence_score(["hit_summary", "hit_keywords"]),
            EVIDENCE_WEIGHTS["hit_keywords"],
        )


class LexicalRescueKeepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.restricted = build_search_policy("테스트")
        self.normal = build_search_policy("무선충전기")

    def test_legacy_when_rescue_off(self) -> None:
        keep, reason = lexical_rescue_keep(["hit_summary"], policy=self.restricted, rescue_enabled=False)
        self.assertTrue(keep)
        self.assertEqual(reason, "legacy_lexical")

    def test_weak_only_restricted_drop(self) -> None:
        keep, reason = lexical_rescue_keep(
            ["hit_summary", "hit_search_text"],
            policy=self.restricted,
            rescue_enabled=True,
        )
        self.assertFalse(keep)
        self.assertEqual(reason, "dropped_weak")

    def test_strong_restricted_keep(self) -> None:
        keep, reason = lexical_rescue_keep(
            ["hit_keywords"],
            policy=self.restricted,
            rescue_enabled=True,
        )
        self.assertTrue(keep)
        self.assertEqual(reason, "evidence_restricted")

    def test_normal_policy_keywords_keep(self) -> None:
        keep, _ = lexical_rescue_keep(
            ["hit_keywords"],
            policy=self.normal,
            rescue_enabled=True,
        )
        self.assertTrue(keep)

    def test_keyword_mode_weak_keep(self) -> None:
        policy = build_search_policy("테스트", mode="keyword")
        keep, reason = lexical_rescue_keep(
            ["hit_summary", "hit_search_text"],
            policy=policy,
            rescue_enabled=True,
        )
        self.assertTrue(keep)
        self.assertEqual(reason, "evidence_keyword")


if __name__ == "__main__":
    unittest.main()
