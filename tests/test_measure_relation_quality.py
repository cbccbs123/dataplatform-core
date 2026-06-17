"""spec 033 T008 — measure 러너 순수 단위.

- build_snapshot 이 제안 엣지에 후보 emb_score 를 부착해 ProposedEdge.emb_score 로 동결하는지(FR-006).
- measure 커맨드(cmd_measure)가 동결 스냅샷 위에서 min_sim_sweep·auto_approve_sweep 표를
  리포트에 합쳐 출력하는지(읽기 전용·graph_edge 무기록 유지).

DB/LLM 불요 — db.transaction·resolve_asset_keys·_read_candidates_prompt 를 모킹하고 llm_fn 주입.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from unittest import mock


@contextlib.contextmanager
def _fake_tx():
    """db.transaction() 컨텍스트 매니저 스텁 — conn 으로 MagicMock 을 내준다(쿼리는 패치)."""
    yield mock.MagicMock()


class TestBuildSnapshotEmbScore(unittest.TestCase):
    """build_snapshot 이 후보 emb_score 를 ProposedEdge.emb_score 로 동결하는지(FR-006)."""

    def test_proposed_edge_gets_candidate_emb_score(self) -> None:
        from scripts import measure_relation_quality as mrq

        from src.relations.quality.golden import parse_golden

        # 골든: 소스 키 src/dst → asset_id 해소(모킹).
        golden = parse_golden({
            "version": 1, "key_type": "fs_path",
            "pairs": [{"a": "src", "b": "dst", "kind": "same_domain"}],
            "isolated": [],
        })
        mapping = {"src": "id_src", "dst": "id_dst"}
        # 후보 스텁: dst 는 emb 후보(0.73), other 는 path-only(0.0).
        cands = [
            {"id": "id_dst", "emb_score": 0.73},
            {"id": "id_other", "emb_score": 0.0},
        ]
        # LLM 주입: dst·other 두 엣지 제안.
        def fake_llm(_p):
            return {"edges": [
                {"target_media_item_id": "id_dst", "relation_type_code": "same_domain", "confidence": 0.9},
                {"target_media_item_id": "id_other", "relation_type_code": "same_domain", "confidence": 0.8},
            ]}

        db = mock.MagicMock()
        db.transaction.side_effect = lambda: _fake_tx()
        with mock.patch.object(mrq, "resolve_asset_keys", return_value=(mapping, [])), \
             mock.patch.object(mrq, "_read_candidates_prompt", return_value=(cands, "PROMPT")), \
             mock.patch.object(mrq, "_settings", return_value=mock.MagicMock()):
            snap, mp, missing = mrq.build_snapshot(
                db, golden, config={"top_k": 10, "min_sim": 0.0, "embedding_kind": "both"},
                llm_fn=fake_llm)

        self.assertEqual(missing, [])
        # 소스 id_src 의 제안 엣지에 후보 emb_score 가 부착되어야 한다.
        edges = {e.target: e for ss in snap.sources.values() for e in ss.proposed}
        self.assertAlmostEqual(edges["id_dst"].emb_score, 0.73)
        # 후보 맵에 없는 타깃이면 0.0(sentinel). other 는 후보에 0.0 으로 존재.
        self.assertAlmostEqual(edges["id_other"].emb_score, 0.0)

    def test_proposed_edge_unknown_target_defaults_zero(self) -> None:
        from scripts import measure_relation_quality as mrq

        from src.relations.quality.golden import parse_golden

        golden = parse_golden({
            "version": 1, "key_type": "fs_path",
            "pairs": [{"a": "src", "b": "dst", "kind": "same_domain"}],
            "isolated": [],
        })
        mapping = {"src": "id_src", "dst": "id_dst"}
        cands = [{"id": "id_dst", "emb_score": 0.55}]
        # LLM 이 후보 맵에 없는 타깃(환각)을 제안 — emb_score 는 0.0 으로 동결.
        def fake_llm(_p):
            return {"edges": [
                {"target_media_item_id": "id_ghost", "relation_type_code": "same_domain", "confidence": 0.9},
            ]}
        db = mock.MagicMock()
        db.transaction.side_effect = lambda: _fake_tx()
        with mock.patch.object(mrq, "resolve_asset_keys", return_value=(mapping, [])), \
             mock.patch.object(mrq, "_read_candidates_prompt", return_value=(cands, "PROMPT")), \
             mock.patch.object(mrq, "_settings", return_value=mock.MagicMock()):
            snap, _mp, _missing = mrq.build_snapshot(
                db, golden, config={"top_k": 10, "min_sim": 0.0, "embedding_kind": "both"},
                llm_fn=fake_llm)
        edges = {e.target: e for ss in snap.sources.values() for e in ss.proposed}
        self.assertAlmostEqual(edges["id_ghost"].emb_score, 0.0)


class TestCmdMeasureSweeps(unittest.TestCase):
    """cmd_measure 가 동결 스냅샷 위에서 min_sim·auto_approve 2D 스윕 표를 리포트에 합치는지."""

    def _write_snapshot(self, payload: dict) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, f, ensure_ascii=False)
        f.close()
        return f.name

    def test_measure_includes_sweep_tables(self) -> None:
        from scripts.measure_relation_quality import cmd_measure

        from src.relations.quality.golden import parse_golden
        from src.relations.quality.snapshot import (
            ProposedEdge,
            Snapshot,
            SourceSnapshot,
            dump_snapshot,
        )

        # 골든: id_src-id_dst 양성 1쌍. 제안: dst(true, emb 0.40), other(false, emb 0.80).
        golden = parse_golden({
            "version": 1, "key_type": "fs_path",
            "pairs": [{"a": "src", "b": "dst", "kind": "k"}],
            "isolated": [],
        })
        key_to_id = {"src": "id_src", "dst": "id_dst"}
        snap = Snapshot(config={}, sources={
            "id_src": SourceSnapshot(
                # 후보 (id, emb_score) 동결 — N1 스윕이 후보 단계 recall 을 점수 임계로 잰다(FR-004).
                candidates=(("id_dst", 0.40), ("id_other", 0.80)),
                proposed=(
                    ProposedEdge("id_dst", "k", 0.95, "", emb_score=0.40),
                    ProposedEdge("id_other", "k", 0.95, "", emb_score=0.80),
                )),
        })
        snap_path = self._write_snapshot(
            {"snapshot": dump_snapshot(snap), "key_to_id": key_to_id})
        try:
            report = cmd_measure(golden, snap_path)
        finally:
            os.unlink(snap_path)

        json.dumps(report)  # 직렬화 가능
        # 신규 스윕 표가 리포트에 포함된다.
        self.assertIn("min_sim_sweep", report)
        self.assertIn("auto_approve_sweep", report)
        self.assertIsInstance(report["min_sim_sweep"], list)
        self.assertIsInstance(report["auto_approve_sweep"], list)
        self.assertTrue(report["min_sim_sweep"])
        self.assertTrue(report["auto_approve_sweep"])

        # 2D 스윕 의미: emb_min=0.0 격자는 둘 다 승인 → precision 0.5(dst true·other false).
        grid = {(r["conf_min"], r["emb_min"]): r for r in report["auto_approve_sweep"]}
        cell0 = grid[(0.9, 0.0)]
        self.assertEqual(cell0["approved"], 2)
        self.assertAlmostEqual(cell0["precision"], 0.5)
        # emb_min=0.5 격자는 dst(0.40) 탈락, other(0.80)만 승인(false) → approved 1·precision 0.0.
        cell5 = grid[(0.9, 0.5)]
        self.assertEqual(cell5["approved"], 1)
        self.assertAlmostEqual(cell5["precision"], 0.0)

        # min_sim 스윕: 동결 후보 (id, emb_score) 기준 recall 표(낮은 하한일수록 recall↑).
        by_sim = {r["min_sim"]: r for r in report["min_sim_sweep"]}
        # 하한 0.0 에서는 후보 dst(emb 0.40)가 살아 골든 쌍 recall=1.0.
        self.assertAlmostEqual(by_sim[0.0]["recall"], 1.0)
        # 하한 0.5 에서는 후보 dst(0.40) 탈락 → recall 0.0.
        self.assertAlmostEqual(by_sim[0.5]["recall"], 0.0)

    def test_measure_is_read_only_no_db(self) -> None:
        # cmd_measure 는 snapshot 파일만 읽는다 — DB/LLM 인자 없음(읽기 전용 시그니처 보존).
        import inspect

        from scripts.measure_relation_quality import cmd_measure
        params = list(inspect.signature(cmd_measure).parameters)
        self.assertEqual(params, ["golden", "snapshot_path"])


if __name__ == "__main__":
    unittest.main()
