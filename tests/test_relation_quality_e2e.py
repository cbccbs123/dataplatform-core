"""spec 031 T010 — 관계 품질 하니스 실 DB e2e 베이스라인 (SC-001).

RUN_DB_E2E 게이트(실 PostgreSQL 필요). LLM은 **fake llm_fn 주입**으로 미호출 → 결정적·LLM 0.
검증: build_snapshot(실 DB 후보 조회) → 파일 round-trip → cmd_measure → 리포트가 한 번에 산출되고
형태·범위가 유효함(SC-001). 동작 변경 0 — graph_edge 무기록(측정 전용·SC-004).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

_RUN_DB_E2E = os.environ.get("RUN_DB_E2E")


@unittest.skipUnless(_RUN_DB_E2E, "RUN_DB_E2E 미설정 — 실 DB e2e skip")
class TestRelationQualityE2E(unittest.TestCase):
    """실 DB 후보 + fake LLM 으로 snapshot→measure 전 경로 스모크(SC-001)."""

    @classmethod
    def setUpClass(cls) -> None:
        from dotenv import load_dotenv

        from src.config.settings import init_settings
        from src.database.postgres_util import PostgresUtil

        dp = Path(__file__).resolve().parents[1] / ".env.dev"
        if dp.is_file():
            load_dotenv(dotenv_path=dp, override=False)
        init_settings("dev")
        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db.__exit__(None, None, None)

    def _registered_fs_paths(self, n: int) -> list[str]:
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT fs_path FROM asset WHERE status = 'registered' AND fs_path IS NOT NULL "
                "ORDER BY asset_id LIMIT %s",
                (n,),
            )
            return [str(r[0]) for r in cur.fetchall()]

    def test_snapshot_measure_baseline(self) -> None:
        from scripts.measure_relation_quality import build_snapshot, cmd_measure

        from src.relations.quality.golden import parse_golden
        from src.relations.quality.snapshot import dump_snapshot

        paths = self._registered_fs_paths(3)
        if len(paths) < 3:
            self.skipTest("registered 자산 3건 미만 — 베이스라인 측정 불가")
        golden = parse_golden({
            "version": 1, "key_type": "fs_path",
            "pairs": [{"a": paths[0], "b": paths[1], "kind": "same_domain"}],
            "isolated": [paths[2]],
        })
        # fake LLM — 제안 0(결정적·실 LLM 미호출). 후보 조회는 실 DB.
        config = {"top_k": 10, "min_sim": 0.0, "embedding_kind": "both"}
        snap, mapping, missing = build_snapshot(
            self.db, golden, config=config, llm_fn=lambda _prompt: {"edges": []})

        self.assertEqual(missing, [])                  # 3 키 모두 asset_id 해소
        self.assertEqual(len(snap.sources), 3)         # pair 양쪽 2 + 고립 1

        # 파일 round-trip(key_to_id 포함) → cmd_measure(LLM/DB 불요)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"snapshot": dump_snapshot(snap), "key_to_id": mapping}, f, ensure_ascii=False)
            snap_path = f.name
        try:
            report = cmd_measure(golden, snap_path)
        finally:
            os.unlink(snap_path)

        json.dumps(report)  # 리포트는 직렬화 가능
        for k in ("candidate_recall", "relation_metrics", "sweep", "unresolved_keys",
                  "n_golden_pairs", "n_isolated"):
            self.assertIn(k, report)
        self.assertEqual(report["unresolved_keys"], [])
        self.assertEqual(report["n_golden_pairs"], 1)
        self.assertEqual(report["n_isolated"], 1)
        self.assertGreaterEqual(report["candidate_recall"], 0.0)
        self.assertLessEqual(report["candidate_recall"], 1.0)
        rm = report["relation_metrics"]
        self.assertEqual(rm["precision"], 0.0)         # 제안 0 → precision 0
        self.assertEqual(rm["isolation_accuracy"], 1.0)  # 고립 자산 엣지 0 → 정확
        self.assertIsInstance(report["sweep"], list)
        self.assertTrue(report["sweep"])               # 임계 격자 비어있지 않음


if __name__ == "__main__":
    unittest.main()
