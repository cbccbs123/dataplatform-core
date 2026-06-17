"""관계 품질 메트릭 순수 단위테스트 (spec 031 T003·T004·T005).

LLM/DB 불요 — 후보 recall(대칭 인정)·관계 P/R·kind/고립 정확도·임계 스윕을
합성 입력으로 전수 검증한다(SC-003). 대칭 kind는 양방향 인정(헌법 3조).
"""
import unittest

from src.relations.quality.metrics import (
    auto_approve_sweep,
    candidate_recall,
    min_sim_sweep,
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


class TestMinSimSweep(unittest.TestCase):
    """033 T004(FR-004): min_sim 하한 스윕 — 각 하한에서 emb_score>=t 후보만 남기고 recall·후보수."""

    def test_min_sim_sweep_monotone(self):
        # 소스 src1 의 후보: (dst1, 0.8), (dst2, 0.3); 골든 양성 {src1-dst1, src1-dst2}
        cands = {"src1": [("dst1", 0.8), ("dst2", 0.3)]}
        golden = [("src1", "dst1"), ("src1", "dst2")]
        rows = min_sim_sweep(golden, cands, thresholds=[0.2, 0.5, 0.9])
        by = {r["min_sim"]: r for r in rows}
        self.assertEqual(by[0.2]["recall"], 1.0)     # 둘 다 통과
        self.assertEqual(by[0.5]["recall"], 0.5)     # dst2 탈락
        self.assertEqual(by[0.9]["recall"], 0.0)
        # 후보 수도 단조 감소
        self.assertGreaterEqual(by[0.2]["candidates"], by[0.5]["candidates"])
        self.assertGreaterEqual(by[0.5]["candidates"], by[0.9]["candidates"])
        self.assertEqual(by[0.2]["candidates"], 2)
        self.assertEqual(by[0.5]["candidates"], 1)
        self.assertEqual(by[0.9]["candidates"], 0)

    def test_min_sim_sweep_symmetric_recall(self):
        # 대칭 인정: 골든 (a,b)는 b∈cand[a] 또는 a∈cand[b] 면 회수.
        cands = {"a": [], "b": [("a", 0.7)]}
        rows = min_sim_sweep([("a", "b")], cands, thresholds=[0.5, 0.8])
        by = {r["min_sim"]: r for r in rows}
        self.assertEqual(by[0.5]["recall"], 1.0)  # b→a 역방향으로 회수
        self.assertEqual(by[0.8]["recall"], 0.0)  # 0.7 탈락

    def test_min_sim_sweep_preserves_threshold_order(self):
        rows = min_sim_sweep([], {}, thresholds=[0.2, 0.5, 0.9])
        self.assertEqual([r["min_sim"] for r in rows], [0.2, 0.5, 0.9])


class TestAutoApproveSweep(unittest.TestCase):
    """033 T005(FR-005): 2D(conf×emb) 자동승인 스윕 — precision·승인 수."""

    def test_auto_approve_sweep_and_semantics(self):
        proposed = {"src1": [ProposedEdge("dst1", "k", 0.95, emb_score=0.40),   # 골든 true
                             ProposedEdge("dst2", "k", 0.95, emb_score=0.80)]}  # 골든 false
        golden = [("src1", "dst1")]
        grid = auto_approve_sweep(golden, proposed, conf_thresholds=[0.9], emb_thresholds=[0.0, 0.5])
        g = {(c["conf_min"], c["emb_min"]): c for c in grid}
        # emb_min=0.0: 둘 다 자동승인 → precision 0.5
        self.assertAlmostEqual(g[(0.9, 0.0)]["precision"], 0.5)
        self.assertEqual(g[(0.9, 0.0)]["approved"], 2)
        # emb_min=0.5: dst1(0.40) 탈락, dst2(0.80)만 승인(골든 false) → precision 0.0, approved 1
        self.assertEqual(g[(0.9, 0.5)]["approved"], 1)
        self.assertAlmostEqual(g[(0.9, 0.5)]["precision"], 0.0)

    def test_auto_approve_sweep_conf_gate(self):
        # conf_min 이 conf 보다 높으면 승인 0 → precision 0.0(승인 없음 관례).
        proposed = {"s": [ProposedEdge("d", "k", 0.80, emb_score=0.9)]}
        grid = auto_approve_sweep([("s", "d")], proposed, conf_thresholds=[0.9], emb_thresholds=[0.0])
        self.assertEqual(grid[0]["approved"], 0)
        self.assertEqual(grid[0]["precision"], 0.0)

    def test_auto_approve_sweep_grid_shape(self):
        proposed = {"s": [ProposedEdge("d", "k", 0.95, emb_score=0.5)]}
        grid = auto_approve_sweep([("s", "d")], proposed,
                                  conf_thresholds=[0.8, 0.9], emb_thresholds=[0.0, 0.5])
        self.assertEqual(len(grid), 4)  # 2×2
        keys = {(c["conf_min"], c["emb_min"]) for c in grid}
        self.assertEqual(keys, {(0.8, 0.0), (0.8, 0.5), (0.9, 0.0), (0.9, 0.5)})


if __name__ == "__main__":
    unittest.main()
