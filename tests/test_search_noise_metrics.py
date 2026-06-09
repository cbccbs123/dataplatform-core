"""무관(non-relevant) 노이즈 지표 순수 함수 단위 (019 G3 — T008).

검색 랭킹에서 **정답이 아닌 자산**이 얼마나 위로 오는지를 수치화한다. 청크 집계 MAX 는 긴 무관
영상(많은 청크 중 운 좋은 한 청크)을 상위로 끌어올리는 경향이 있어(spec §무엇·왜), 집계 방식
비교 시 recall/MRR/nDCG 만으로는 "무관 영상이 위로 오는 정도"가 직접 드러나지 않는다. 그래서
두 보조 지표를 둔다(둘 다 ranked=id 순위, relevant=정답 id 집합의 순수 함수, 헌법 3조 결정적):

  (a) ``nonrelevant_mean_rank`` — 정답 아닌 자산의 **평균 1-기반 순위**. 높을수록 무관 자산이
      아래로 밀린 것(좋음). 수식: mean({ i : ranked[i-1] ∉ relevant }), i 는 1-기반.
  (b) ``nonrelevant_exposure_at_k`` — 상위 k **노출률** = (상위 k 중 정답 아님) / min(k, len(ranked)).
      낮을수록 좋음(긴 무관 영상이 top-N 을 덜 차지).

경계(정답 0개·전부 정답·빈 랭킹)를 명시적으로 못박는다 — 조용히 잘못된 값을 내지 않도록.
"""

from __future__ import annotations

import unittest

from tests.fixtures.search.metrics import nonrelevant_exposure_at_k, nonrelevant_mean_rank


class TestNonrelevantMeanRank(unittest.TestCase):
    """정답 아닌 자산의 평균 순위(높을수록 무관 자산이 아래로 — 좋음)."""

    def test_typical_ranking(self) -> None:
        # ranked=[d,a,e,b], relevant={a,b} → 무관 d@1·e@3 → 평균 (1+3)/2 = 2.0
        self.assertAlmostEqual(nonrelevant_mean_rank(["d", "a", "e", "b"], {"a", "b"}), 2.0)

    def test_no_relevant_all_nonrelevant(self) -> None:
        # 정답 0개 → 전 항목이 무관 → 평균 = (1+2+3)/3 = 2.0 = (n+1)/2
        self.assertAlmostEqual(nonrelevant_mean_rank(["x", "y", "z"], set()), 2.0)

    def test_all_relevant_returns_zero(self) -> None:
        # 전부 정답 → 무관 항목 없음 → 0.0(측정 대상 없음)
        self.assertEqual(nonrelevant_mean_rank(["a", "b"], {"a", "b", "c"}), 0.0)

    def test_empty_ranking_returns_zero(self) -> None:
        self.assertEqual(nonrelevant_mean_rank([], {"a"}), 0.0)

    def test_deterministic(self) -> None:
        ranked, rel = ["m", "a", "n", "b", "o"], {"a", "b"}
        self.assertEqual(
            nonrelevant_mean_rank(ranked, rel), nonrelevant_mean_rank(ranked, rel)
        )


class TestNonrelevantExposureAtK(unittest.TestCase):
    """상위 k 무관 노출률(낮을수록 좋음). 분모는 실제로 상위 k 에 노출된 항목 수."""

    def test_typical_top_k(self) -> None:
        # ranked=[d,a,e,b], relevant={a,b}, k=4 → 무관 d,e 2개 / 4 = 0.5
        self.assertAlmostEqual(nonrelevant_exposure_at_k(["d", "a", "e", "b"], {"a", "b"}, 4), 0.5)

    def test_k_larger_than_ranking_uses_actual_length(self) -> None:
        # ranked=[e,f,c](3건), k=10 → 분모 min(10,3)=3, 무관 e,f → 2/3
        self.assertAlmostEqual(
            nonrelevant_exposure_at_k(["e", "f", "c"], {"c"}, 10), 2 / 3
        )

    def test_no_relevant_all_noise(self) -> None:
        # 정답 0개 → 상위 k 전부 무관 → 1.0
        self.assertEqual(nonrelevant_exposure_at_k(["x", "y", "z"], set(), 5), 1.0)

    def test_all_relevant_zero_exposure(self) -> None:
        self.assertEqual(nonrelevant_exposure_at_k(["a", "b"], {"a", "b"}, 5), 0.0)

    def test_empty_ranking_returns_zero(self) -> None:
        self.assertEqual(nonrelevant_exposure_at_k([], {"a"}, 5), 0.0)

    def test_non_positive_k_returns_zero(self) -> None:
        # k<=0 은 노출 슬롯 없음 → 0.0(0 나눗셈 방지)
        self.assertEqual(nonrelevant_exposure_at_k(["x", "a"], {"a"}, 0), 0.0)

    def test_only_top_k_counted_not_beyond(self) -> None:
        # 상위 k 밖 무관은 노출률에 안 들어간다. ranked=[a,b,x,y], k=2, rel={a,b} → 0.0
        self.assertEqual(nonrelevant_exposure_at_k(["a", "b", "x", "y"], {"a", "b"}, 2), 0.0)


if __name__ == "__main__":
    unittest.main()
