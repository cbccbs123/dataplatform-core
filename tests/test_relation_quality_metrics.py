"""관계 품질 메트릭 순수 단위테스트 (spec 031 T003·T004·T005).

LLM/DB 불요 — 후보 recall(대칭 인정)·관계 P/R·kind/고립 정확도·임계 스윕을
합성 입력으로 전수 검증한다(SC-003). 대칭 kind는 양방향 인정(헌법 3조).
"""
import unittest

from src.relations.quality.metrics import (
    candidate_recall,
    relation_metrics,
    threshold_sweep,
)
from src.relations.quality.snapshot import ProposedEdge


class TestCandidateRecall(unittest.TestCase):
    def test_symmetric_either_direction(self):
        pairs = [("a", "b"), ("c", "d")]
        # a→b 직접, d→c 역방향 — 대칭이므로 둘 다 회수.
        cands = {"a": {"b"}, "b": set(), "c": set(), "d": {"c"}}
        self.assertEqual(candidate_recall(pairs, cands), 1.0)

    def test_missed_pair(self):
        pairs = [("a", "b")]
        cands = {"a": {"z"}, "b": {"y"}}
        self.assertEqual(candidate_recall(pairs, cands), 0.0)

    def test_empty_pairs(self):
        self.assertEqual(candidate_recall([], {}), 0.0)


class TestRelationMetrics(unittest.TestCase):
    def _proposed(self):
        return {"a": [ProposedEdge("b", "same_series", 0.9),
                      ProposedEdge("z", "same_domain", 0.4)]}

    def test_precision_recall_kind(self):
        m = relation_metrics(
            triples=[("a", "b", "same_series")], isolated=set(),
            proposed=self._proposed(), confidence_min=0.0)
        self.assertEqual(m["recall"], 1.0)         # 골든 (a,b) 회수
        self.assertEqual(m["precision"], 0.5)      # 2엣지 중 (a,b)만 정답
        self.assertEqual(m["kind_accuracy"], 1.0)  # 매칭된 (a,b) kind 일치

    def test_confidence_min_filters(self):
        m = relation_metrics(
            triples=[("a", "b", "same_series")], isolated=set(),
            proposed=self._proposed(), confidence_min=0.5)  # 0.4 엣지 탈락
        self.assertEqual(m["precision"], 1.0)

    def test_isolation_accuracy(self):
        m = relation_metrics(
            triples=[], isolated={"iso1", "iso2"},
            proposed={"iso1": [ProposedEdge("x", "same_domain", 0.9)]},
            confidence_min=0.0)
        self.assertEqual(m["isolation_accuracy"], 0.5)  # iso2만 엣지0


class TestSweep(unittest.TestCase):
    def test_sweep_monotone(self):
        proposed = {"a": [ProposedEdge("b", "same_series", 0.9),
                          ProposedEdge("z", "x", 0.4)]}
        rows = threshold_sweep(
            triples=[("a", "b", "same_series")], isolated=set(),
            proposed=proposed, thresholds=[0.0, 0.5, 0.95])
        self.assertEqual([r["confidence_min"] for r in rows], [0.0, 0.5, 0.95])
        self.assertEqual(rows[0]["precision"], 0.5)  # 0.0: 2엣지
        self.assertEqual(rows[1]["precision"], 1.0)  # 0.5: (a,b)만
        self.assertEqual(rows[2]["recall"], 0.0)     # 0.95: 아무 엣지도 통과 못함


if __name__ == "__main__":
    unittest.main()
