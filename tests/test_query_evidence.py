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
        # 설계 §11.2 예: summary + cross_meta = 0.7 + 0.3
        self.assertAlmostEqual(
            evidence_score(["hit_summary", "hit_cross_meta"]),
            1.0,
        )

    def test_keywords_only(self) -> None:
        self.assertAlmostEqual(evidence_score(["hit_keywords"]), 3.0)

    def test_dedup_cross_meta_when_strong(self) -> None:
        # keywords hit 시 cross_meta 중복 가산 금지
        score = evidence_score(["hit_keywords", "hit_cross_meta"])
        self.assertAlmostEqual(score, 3.0)

    def test_strong_evidence_score(self) -> None:
        self.assertAlmostEqual(
            strong_evidence_score(["hit_summary", "hit_keywords"]),
            EVIDENCE_WEIGHTS["hit_keywords"],
        )


class LexicalRescueKeepTest(unittest.TestCase):
    """2026-07-24 mode 슬림: restricted 분기 제거 — auto=NORMAL(1.5) / keyword=KEYWORD(0.7) 임계만."""

    def setUp(self) -> None:
        self.auto = build_search_policy("무선충전기")

    def test_legacy_when_rescue_off(self) -> None:
        keep, reason = lexical_rescue_keep(["hit_summary"], policy=self.auto, rescue_enabled=False)
        self.assertTrue(keep)
        self.assertEqual(reason, "legacy_lexical")

    def test_weak_auto_drop(self) -> None:
        # summary+cross_meta = 1.0 < NORMAL 1.5 → drop.
        keep, reason = lexical_rescue_keep(
            ["hit_summary", "hit_cross_meta"], policy=self.auto, rescue_enabled=True)
        self.assertFalse(keep)
        self.assertEqual(reason, "dropped_weak")

    def test_normal_keywords_keep(self) -> None:
        # keywords 3.0 ≥ NORMAL 1.5 → keep.
        keep, reason = lexical_rescue_keep(["hit_keywords"], policy=self.auto, rescue_enabled=True)
        self.assertTrue(keep)
        self.assertEqual(reason, "evidence_normal")

    def test_keyword_mode_weak_keep(self) -> None:
        # keyword 모드는 관대한 하한(0.7) — weak(1.0)도 keep.
        policy = build_search_policy("테스트", mode="keyword")
        keep, reason = lexical_rescue_keep(
            ["hit_summary", "hit_cross_meta"], policy=policy, rescue_enabled=True)
        self.assertTrue(keep)
        self.assertEqual(reason, "evidence_keyword")


if __name__ == "__main__":
    unittest.main()
