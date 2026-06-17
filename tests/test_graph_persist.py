"""v2 단계 C — 공용 그래프 영속화 테스트.

- TestSyncGraphEdgesUnit: mock conn 으로 필터 로직 검증(DB 불필요).
- Test*DB: RUN_DB_E2E=1 일 때만, 실 PostgreSQL 에 node/graph_edge 적재 검증.
"""
from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest import mock

from dotenv import load_dotenv

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"

_SRC = "018f0000-0000-7000-8000-000000000001"
_T1 = "018f0000-0000-7000-8000-000000000007"
_T2 = "018f0000-0000-7000-8000-000000000008"


# ── 순수 단위(mock conn) — DB 불필요 ────────────────────────────────────────
class TestCanonicalOrdering(unittest.TestCase):
    def test_symmetric_kind_orders_pair(self):
        from src.relations.graph_persist import _canonical_pair
        a, b = "018f0000-0000-7000-8000-000000000006", "018f0000-0000-7000-8000-000000000005"
        # 대칭: 항상 (min, max)
        self.assertEqual(_canonical_pair(a, b, symmetric=True), (b, a))
        self.assertEqual(_canonical_pair(b, a, symmetric=True), (b, a))
        # 비대칭: 입력 방향 유지
        self.assertEqual(_canonical_pair(a, b, symmetric=False), (a, b))


class TestSyncGraphEdgesUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = mock.MagicMock()
        self.cur = self.conn.cursor.return_value.__enter__.return_value
        self.allowed = frozenset({_T1, _T2})

    def _edge(self, **kw):
        e = {"target_media_item_id": _T1, "relation_type_code": "same_domain", "reason": "유사"}
        e.update(kw)
        return e

    def _run(self, edges, kind=("k1", True)):
        """kind=(relation_kind_id, is_symmetric) 또는 None(미해소 kind)."""
        from src.relations import graph_persist
        kdict = None if kind is None else {"relation_kind_id": kind[0], "is_symmetric": kind[1]}
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind", return_value=kdict):
            return graph_persist.sync_graph_edges(
                self.conn, source_asset_id=_SRC, edges=edges, allowed_target_ids=self.allowed
            )

    def test_valid_edge_upserted(self) -> None:
        up, sk = self._run([self._edge()])
        self.assertEqual((up, sk), (1, 0))
        sql = " ".join(str(c.args[0]) for c in self.cur.execute.call_args_list)
        self.assertIn("INSERT INTO graph_edge", sql)

    def _insert_params(self, edges, kind=("k1", True), **kwargs):
        """단일 엣지 upsert 의 INSERT 바인딩 파라미터(ensure_asset_node mock — INSERT 만 self.cur 사용)."""
        from src.relations import graph_persist
        kdict = {"relation_kind_id": kind[0], "is_symmetric": kind[1]}
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind", return_value=kdict):
            graph_persist.sync_graph_edges(
                self.conn, source_asset_id=_SRC, edges=edges, allowed_target_ids=self.allowed, **kwargs)
        return self.cur.execute.call_args[0][1]

    def test_status_proposed_below_auto_approve(self) -> None:
        params = self._insert_params([self._edge(confidence=0.5)], auto_approve_min=0.9)
        self.assertEqual(params[-1], "proposed")  # status_val 은 INSERT 마지막 바인딩

    def test_status_active_at_or_above_auto_approve(self) -> None:
        params = self._insert_params([self._edge(confidence=0.95)], auto_approve_min=0.9)
        self.assertEqual(params[-1], "active")

    def test_topic_jsonb_serialized(self) -> None:
        import json
        params = self._insert_params([self._edge(topic_ko="게임", topic_en="gaming")])
        topic = json.loads(params[6])  # (edge_id,a,b,kind,conf,reason,topic,status) 중 7번째
        self.assertEqual(topic["topic_ko"], "게임")
        self.assertEqual(topic["topic_en"], "gaming")

    def test_target_not_in_allowed_skipped(self) -> None:
        self.assertEqual(self._run([self._edge(target_media_item_id="018f0000-0000-7000-8000-000000000014")]), (0, 1))

    def test_invalid_uuid_target_skipped(self) -> None:
        self.assertEqual(self._run([self._edge(target_media_item_id="42")]), (0, 1))

    def test_self_reference_skipped(self) -> None:
        self.assertEqual(self._run([self._edge(target_media_item_id=_SRC)]), (0, 1))

    def test_missing_relation_code_skipped(self) -> None:
        self.assertEqual(self._run([self._edge(relation_type_code="")]), (0, 1))

    def test_unresolved_relation_kind_skipped(self) -> None:
        self.assertEqual(self._run([self._edge()], kind=None), (0, 1))


# ── 실 DB 통합(RUN_DB_E2E=1) ────────────────────────────────────────────────
def _vec():
    from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
    v = [0.0] * FIX_EMBEDDING_DIMENSION
    v[0] = 0.5  # 비영 벡터(코사인 정의)
    return v


def _make_registered_asset(db, ids: list) -> str:
    """registered + st 임베딩 보유 자산 1건 생성(테스트 헬퍼). 생성 id 를 ids 에 누적."""
    from src.dispatch.types import AssetRecord, EmbeddingItem
    from src.ingest.status import AssetStatus, set_status
    from src.registry.asset_persist import create_asset, finalize_asset

    with db.transaction() as conn:
        aid = create_asset(conn, fs_path=f"/t/{uuid.uuid4().hex}.txt", modality="txt", file_hash=uuid.uuid4().hex)
    ids.append(aid)
    with db.transaction() as conn:
        set_status(conn, aid, AssetStatus.ROUTING)
        set_status(conn, aid, AssetStatus.CLASSIFYING)
        set_status(conn, aid, AssetStatus.EXTRACTING)
    with db.transaction() as conn:
        finalize_asset(conn, aid, AssetRecord(
            fts_plain="x", embeddings=[EmbeddingItem(channel="st", vector=_vec(), model_name="m")]))
    return str(aid)


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만")
class TestGraphPersistDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(_ENV, override=False)
        from src.database.postgres_util import PostgresUtil
        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.db.__exit__(None, None, None)

    def setUp(self):
        self._ids: list = []

    def tearDown(self):
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def test_schema_exists(self):
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.node'), to_regclass('public.graph_edge')")
            n, e = cur.fetchone()
        self.assertIsNotNone(n)
        self.assertIsNotNone(e)

    def test_sync_graph_edges_creates_nodes_and_edge(self):
        from src.relations.graph_persist import sync_graph_edges
        src_id = _make_registered_asset(self.db, self._ids)
        dst_id = _make_registered_asset(self.db, self._ids)
        edges = [{
            "target_media_item_id": dst_id, "relation_type_code": "same_domain",
            "topic_ko": "일반", "topic_en": "general", "subtopic_ko": "", "subtopic_en": "",
            "confidence": 0.8, "reason": "테스트",
        }]

        def _run(conn):
            return sync_graph_edges(conn, source_asset_id=src_id, edges=edges, allowed_target_ids=frozenset({dst_id}))

        up, sk = self.db.execute_in_transaction(_run, idempotent=False)
        self.assertEqual((up, sk), (1, 0))
        # 멱등 — 같은 엣지 재실행
        up2, _ = self.db.execute_in_transaction(_run, idempotent=False)
        self.assertEqual(up2, 1)
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) FROM graph_edge ge
                JOIN node sn ON sn.node_id = ge.src_node AND sn.asset_id = %s
                JOIN node dn ON dn.node_id = ge.dst_node AND dn.asset_id = %s
            """, (src_id, dst_id))
            (n,) = cur.fetchone()
        self.assertEqual(n, 1)   # 멱등: 1건 유지

    # ── 032: 충돌 시 confidence 더 큰 제안의 topic·reason 갱신(status 보존) ──
    def _sync_edge(self, src_id, dst_id, *, confidence, topic_ko, reason):
        """단일 derived_from(비대칭 — canonical 방향 유지) 엣지 sync. (upserted, skipped) 반환."""
        from src.relations.graph_persist import sync_graph_edges
        edges = [{
            "target_media_item_id": dst_id, "relation_type_code": "derived_from",
            "topic_ko": topic_ko, "topic_en": "", "subtopic_ko": "", "subtopic_en": "",
            "confidence": confidence, "reason": reason,
        }]
        return self.db.execute_in_transaction(
            lambda conn: sync_graph_edges(
                conn, source_asset_id=src_id, edges=edges, allowed_target_ids=frozenset({dst_id})),
            idempotent=False)

    def _edge_row(self, src_id, dst_id):
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT ge.topic->>'topic_ko', ge.reason, ge.confidence, ge.status FROM graph_edge ge "
                "JOIN node sn ON sn.node_id = ge.src_node AND sn.asset_id = %s "
                "JOIN node dn ON dn.node_id = ge.dst_node AND dn.asset_id = %s",
                (src_id, dst_id))
            return cur.fetchone()

    def test_conflict_refreshes_topic_when_higher_confidence(self):
        src_id = _make_registered_asset(self.db, self._ids)
        dst_id = _make_registered_asset(self.db, self._ids)
        self._sync_edge(src_id, dst_id, confidence=0.5, topic_ko="기타", reason="r1")
        self._sync_edge(src_id, dst_id, confidence=0.9, topic_ko="게임", reason="r2")  # 더 높음 → 갱신
        topic, reason, conf, _ = self._edge_row(src_id, dst_id)
        self.assertEqual(topic, "게임")
        self.assertEqual(reason, "r2")
        self.assertEqual(float(conf), 0.9)
        self._sync_edge(src_id, dst_id, confidence=0.3, topic_ko="잡담", reason="r3")  # 더 낮음 → 보존
        topic2, reason2, conf2, _ = self._edge_row(src_id, dst_id)
        self.assertEqual(topic2, "게임")            # 보존
        self.assertEqual(reason2, "r2")
        self.assertEqual(float(conf2), 0.9)         # GREATEST
        self._sync_edge(src_id, dst_id, confidence=0.9, topic_ko="동률", reason="r4")  # 동률(==) → strict > 라 보존
        topic3, reason3, _, _ = self._edge_row(src_id, dst_id)
        self.assertEqual(topic3, "게임")            # 동률은 갱신 안 함(strict > 경계·결정적)
        self.assertEqual(reason3, "r2")

    def test_conflict_preserves_status_even_higher_confidence(self):
        src_id = _make_registered_asset(self.db, self._ids)
        dst_id = _make_registered_asset(self.db, self._ids)
        self._sync_edge(src_id, dst_id, confidence=0.5, topic_ko="기타", reason="r1")
        with self.db.transaction() as conn, conn.cursor() as cur:  # 사람이 rejected 처리
            cur.execute(
                "UPDATE graph_edge ge SET status='rejected' FROM node sn, node dn "
                "WHERE sn.node_id=ge.src_node AND sn.asset_id=%s "
                "AND dn.node_id=ge.dst_node AND dn.asset_id=%s",
                (src_id, dst_id))
        self._sync_edge(src_id, dst_id, confidence=0.95, topic_ko="게임", reason="r2")  # 높아도
        _, _, _, status = self._edge_row(src_id, dst_id)
        self.assertEqual(status, "rejected")        # status 보존(사람 결정 무손상)


if __name__ == "__main__":
    unittest.main()
