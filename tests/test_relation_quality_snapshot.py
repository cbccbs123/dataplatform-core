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


if __name__ == "__main__":
    unittest.main()
