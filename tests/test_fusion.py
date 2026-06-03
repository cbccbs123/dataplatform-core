"""RRF 융합 순수 함수 단위 테스트(DB·LLM 불요)."""
from __future__ import annotations

import inspect
import unittest

from src.search import media_search, search_service
from src.search.fusion import RRF_DEFAULT_K, fuse_rrf
from src.search.media_search import _apply_fusion


class TestFuseRRF(unittest.TestCase):
    def test_both_lists_top_wins(self) -> None:
        # A는 두 리스트 모두 상위 → 1위
        out = fuse_rrf([["A", "B", "C"], ["A", "C", "B"]], k=60)
        self.assertEqual(out[0][0], "A")

    def test_tiebreak_by_id_ascending(self) -> None:
        # A는 list1 1위/list2 2위, B는 list1 2위/list2 1위 → 점수 동일 → id 오름차순
        out = fuse_rrf([["A", "B"], ["B", "A"]], k=60)
        self.assertEqual([i for i, _ in out], ["A", "B"])
        self.assertAlmostEqual(out[0][1], out[1][1])

    def test_item_in_one_list_ranks_below_consensus(self) -> None:
        # C는 list1에만(2위), A는 두 리스트 모두 → A > C
        out = dict(fuse_rrf([["A", "C"], ["A"]], k=60))
        self.assertGreater(out["A"], out["C"])

    def test_larger_k_compresses_score(self) -> None:
        top_small = fuse_rrf([["A", "B"]], k=1)[0][1]
        top_large = fuse_rrf([["A", "B"]], k=1000)[0][1]
        self.assertGreater(top_small, top_large)

    def test_empty_input(self) -> None:
        self.assertEqual(fuse_rrf([], k=60), [])
        self.assertEqual(fuse_rrf([[]], k=60), [])

    def test_invalid_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            fuse_rrf([["A"]], k=0)

    def test_default_k_is_60(self) -> None:
        self.assertEqual(RRF_DEFAULT_K, 60)


class TestApplyFusion(unittest.TestCase):
    """프로덕션 검색 행 재랭킹 헬퍼(순수). alpha=무변경, rrf=순위 융합 재정렬."""

    def _rows(self):
        # similarity 정렬(alpha 가정)된 행. emb_score 높은 S vs bm25_score 높은 L.
        return [
            {"id": "S", "emb_score": 0.9, "bm25_score": 0.0, "similarity": 0.68},
            {"id": "B", "emb_score": 0.8, "bm25_score": 5.0, "similarity": 0.66},
            {"id": "L", "emb_score": 0.2, "bm25_score": 9.0, "similarity": 0.30},
        ]

    def test_alpha_keeps_order(self) -> None:
        # 동작 불변의 핵심: alpha 면 입력 행을 그대로(순서·내용 유지) 반환
        out = _apply_fusion(self._rows(), fusion="alpha", k=60)
        self.assertEqual([r["id"] for r in out], ["S", "B", "L"])

    def test_rrf_promotes_lexical_consensus(self) -> None:
        # emb순위: S,B,L / bm25순위: L,B,S → B가 양쪽 2위로 최상위권, L은 bm25 1위라 alpha보다↑
        out = _apply_fusion(self._rows(), fusion="rrf", k=60)
        ids = [r["id"] for r in out]
        self.assertEqual(set(ids), {"S", "B", "L"})  # 보존
        self.assertLess(ids.index("L"), 2)  # L이 상위 2위 안으로(alpha에선 3위였음)

    def test_unknown_fusion_raises(self) -> None:
        with self.assertRaises(ValueError):
            _apply_fusion(self._rows(), fusion="zzz", k=60)


class TestFusionWiringDefaults(unittest.TestCase):
    """프로덕션 동작 불변 가드(헌법 8조): 배선된 함수의 fusion 기본값이 모두 alpha 여야 한다.

    검색 진입점들의 기본 경로가 alpha(기존 가중합)로 고정되어 있어야 RRF 배선이 기존
    검색 동작을 바꾸지 않는다(프로토타입은 opt-in).
    """

    def test_defaults_are_alpha(self) -> None:
        for fn in (
            media_search._run_hybrid_search,
            media_search.search_media_all_grouped,
            search_service.search_hybrid,
        ):
            self.assertEqual(
                inspect.signature(fn).parameters["fusion"].default, "alpha"
            )


if __name__ == "__main__":
    unittest.main()
