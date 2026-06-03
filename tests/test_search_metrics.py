"""검색 평가 지표 순수 함수 단위 테스트(DB 불요)."""
from __future__ import annotations

import math
import unittest

from tests.fixtures.search.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.ranked = ["a", "b", "c", "d", "e"]
        self.rel = {"a", "c", "x"}  # x는 미검색(정답 3개 중 2개만 상위)

    def test_recall_at_k(self) -> None:
        self.assertAlmostEqual(recall_at_k(self.ranked, self.rel, 5), 2 / 3)
        self.assertAlmostEqual(recall_at_k(self.ranked, self.rel, 1), 1 / 3)
        self.assertEqual(recall_at_k(self.ranked, set(), 5), 0.0)

    def test_precision_at_k(self) -> None:
        self.assertAlmostEqual(precision_at_k(self.ranked, self.rel, 5), 2 / 5)
        self.assertAlmostEqual(precision_at_k(self.ranked, self.rel, 2), 1 / 2)
        self.assertEqual(precision_at_k(self.ranked, self.rel, 0), 0.0)

    def test_mrr(self) -> None:
        self.assertAlmostEqual(mrr(self.ranked, self.rel), 1.0)  # a가 1위
        self.assertAlmostEqual(mrr(["z", "b", "c"], self.rel), 1 / 3)  # c가 3위
        self.assertEqual(mrr(["z", "y"], self.rel), 0.0)

    def test_ndcg_at_k(self) -> None:
        # 이상 순위: 정답 3개(a,c,x)가 top 에 연속 배치 → 1.0
        self.assertAlmostEqual(ndcg_at_k(["a", "c", "x"], self.rel, 5), 1.0)
        # 정답 3개 중 2개만 회수(x 미검색): 표준 nDCG 는 IDCG 분모를 전체 정답수(3)로 잡아 1.0 미만
        dcg = 1 / math.log2(2) + 1 / math.log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(ndcg_at_k(["a", "c"], self.rel, 5), dcg / idcg)
        # 정답이 아래로 밀리면 더 낮다
        self.assertLess(
            ndcg_at_k(["b", "a", "c"], self.rel, 5), ndcg_at_k(["a", "c"], self.rel, 5)
        )
        self.assertEqual(ndcg_at_k(self.ranked, set(), 5), 0.0)


if __name__ == "__main__":
    unittest.main()
