"""리포트 조립 순수 단위테스트 (spec 031 T008 순수부).

`build_report(golden, snapshot, key_to_id)`는 골든(키 공간)을 스냅샷의 asset_id
공간으로 정합한 뒤 candidate_recall·relation_metrics·threshold_sweep를 한 번에
조립해 리포트 dict를 반환한다(SC-001). LLM/DB 불요 순수 함수 — 합성 입력으로 검증.
"""
import json
import unittest

from src.relations.quality.golden import Golden, GoldenPair
from src.relations.quality.report import build_report
from src.relations.quality.snapshot import ProposedEdge, Snapshot, SourceSnapshot


class TestBuildReport(unittest.TestCase):
    def _golden(self):
        return Golden(
            key_type="fs_path",
            pairs=(GoldenPair("/x/1.mp4", "/x/2.mp4", "same_series", "연작"),),
            isolated=("/x/solo.txt",))

    def _snapshot(self):
        return Snapshot(
            config={"top_k": 10, "min_sim": 0.2, "embedding_kind": "both"},
            sources={
                # candidates 는 정본상 (target_id, emb_score) 튜플(snapshot.py·033 FR-004).
                "id1": SourceSnapshot(
                    candidates=(("id2", 0.8),),
                    proposed=(ProposedEdge("id2", "same_series", 0.9, "강의"),)),
                "id3": SourceSnapshot(candidates=(("id1", 0.7),), proposed=()),
            })

    def _key_to_id(self):
        return {"/x/1.mp4": "id1", "/x/2.mp4": "id2", "/x/solo.txt": "id3"}

    def test_assembles_all_metrics(self):
        report = build_report(self._golden(), self._snapshot(), self._key_to_id())
        json.dumps(report)  # JSON 직렬화 가능(리포트 출력용)
        self.assertEqual(report["candidate_recall"], 1.0)  # id2 ∈ cand[id1]
        self.assertEqual(report["n_golden_pairs"], 1)
        self.assertEqual(report["n_isolated"], 1)
        self.assertEqual(report["unresolved_keys"], [])
        rm = report["relation_metrics"]
        self.assertEqual(rm["precision"], 1.0)
        self.assertEqual(rm["recall"], 1.0)
        self.assertEqual(rm["kind_accuracy"], 1.0)
        self.assertEqual(rm["isolation_accuracy"], 1.0)  # id3 엣지 0 → clean
        self.assertIsInstance(report["sweep"], list)
        self.assertGreater(len(report["sweep"]), 0)
        # 스윕 각 행은 confidence_min + 메트릭 키를 포함.
        self.assertIn("confidence_min", report["sweep"][0])
        self.assertIn("precision", report["sweep"][0])

    def test_confidence_min_threshold(self):
        # confidence_min=0.95 → 0.9 엣지 탈락 → recall 0.
        report = build_report(self._golden(), self._snapshot(), self._key_to_id(),
                              confidence_min=0.95)
        self.assertEqual(report["relation_metrics"]["recall"], 0.0)

    def test_unresolved_key_excluded_and_reported(self):
        golden = Golden(
            key_type="fs_path",
            pairs=(GoldenPair("/x/1.mp4", "/x/missing.mp4", "same_series"),),
            isolated=())
        report = build_report(golden, self._snapshot(), {"/x/1.mp4": "id1"})
        # 한쪽이 미해소 → 쌍 제외, 미해소 키 보고.
        self.assertEqual(report["n_golden_pairs"], 0)
        self.assertIn("/x/missing.mp4", report["unresolved_keys"])
        self.assertEqual(report["candidate_recall"], 0.0)

    def test_custom_thresholds(self):
        report = build_report(self._golden(), self._snapshot(), self._key_to_id(),
                              thresholds=[0.0, 0.95])
        self.assertEqual([r["confidence_min"] for r in report["sweep"]], [0.0, 0.95])


if __name__ == "__main__":
    unittest.main()
