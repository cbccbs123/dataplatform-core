"""LLM 동결 스냅샷 round-trip 순수 단위테스트 (spec 031 T002).

LLM 제안을 스냅샷으로 동결해 재측정·임계 스윕을 LLM 0으로 돌린다(FR-005).
`load_snapshot(dump_snapshot(s)) == s` 와 dump 결과가 JSON 직렬화 가능함을 단언.
"""
import json
import unittest

from src.relations.quality.snapshot import (
    ProposedEdge,
    Snapshot,
    SourceSnapshot,
    dump_snapshot,
    load_snapshot,
)


class TestSnapshot(unittest.TestCase):
    def test_round_trip(self):
        s = Snapshot(
            config={"top_k": 10, "min_sim": 0.2, "embedding_kind": "both"},
            sources={"a1": SourceSnapshot(
                candidates=("a2", "a3"),
                proposed=(ProposedEdge("a2", "same_series", 0.85, "게임"),))})
        d = dump_snapshot(s)
        json.dumps(d)  # 직렬화 가능
        self.assertEqual(load_snapshot(d), s)

    def test_proposededge_emb_score_roundtrip(self):
        # 033 T001: emb_score(후보 코사인 유사도) 동결값이 dump/load 를 왕복해도 보존되는지 (FR-006).
        snap = Snapshot(
            config={},
            sources={"src1": SourceSnapshot(
                candidates=("dst1",),
                proposed=(ProposedEdge("dst1", "same_series", 0.95, "여행", emb_score=0.71),))})
        back = load_snapshot(json.loads(json.dumps(dump_snapshot(snap))))
        self.assertAlmostEqual(back.sources["src1"].proposed[0].emb_score, 0.71)

    def test_legacy_snapshot_without_emb_score_loads_zero(self):
        # 기본값(0.0)이라 emb_score 키 없는 구 스냅샷도 호환 로드(load 시 0.0).
        legacy = {
            "version": 1, "config": {},
            "sources": {"s": {"candidates": ["d"], "proposed": [
                {"target": "d", "kind": "k", "confidence": 0.5, "topic_ko": "x"}]}}}
        snap = load_snapshot(legacy)
        self.assertEqual(snap.sources["s"].proposed[0].emb_score, 0.0)


if __name__ == "__main__":
    unittest.main()
