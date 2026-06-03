"""RRF 융합 순수 함수 단위 테스트(DB·LLM 불요)."""
from __future__ import annotations

import unittest

from src.search.fusion import RRF_DEFAULT_K, fuse_rrf


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


if __name__ == "__main__":
    unittest.main()
