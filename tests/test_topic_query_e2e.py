"""056 G2 — 주제 탐색 seam 실 DB 라운드트립 e2e(RUN_DB_E2E 게이트).

무DB 환경에서는 자동 skip(다른 ``*_e2e`` 관례 일치). ``RUN_DB_E2E=1`` + 로컬
PostgreSQL 에서만 실행한다(사람 게이트 — T203).

검증 의도 (FR-401~403·SC-05·SC-06)
    단위 테스트(``tests/test_topic_query.py``)는 mock cursor 로 파이썬 로직만 본다. 여기서는
    **SQL 자체의 정합**(양끝 자산 조인·active 필터·topic 표현식 술어·의료 제외)을 실 DB 로 검증한다.

시나리오
    소규모 픽스처 그래프(A·B·C 일반, M 의료)를 **충돌 방지 유니크 topic_ko** 로 시딩한다
    (실 dev DB 의 기존 active 엣지 ~수천 건과 겹치지 않게 하여 정확한 카운트 단언 가능).
      엣지(모두 active·유니크 topic): A—B, A—C, B—C, M—B
    - ``find_topic_neighbors(A)``   → {B(overlap2), C(overlap2)} · M 제외 · B/C already_linked
    - ``find_topic_neighbor_groups(A)`` → 주제›하위주제 2단 중첩 {B,C} · M(의료) 제외 · already_linked
    - ``assets_in_topic(topic)``    → {A,B,C} total=3 · M 제외 · 페이징 · subtopic 필터
    - ``list_topics()``             → 유니크 topic 엔트리 asset_count=3 · M 제외
"""
from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

from dotenv import load_dotenv

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만")
class TestTopicQueryDB(unittest.TestCase):
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
        # 충돌 방지 유니크 topic_ko — 실 DB 의 기존 active 엣지와 겹치지 않아 카운트 단언이 견고하다.
        self._topic = "e2e주제_" + uuid.uuid4().hex[:8]
        self._subtopic = "제빵"

    def tearDown(self):
        # asset 삭제 → node(ON DELETE CASCADE) → graph_edge(ON DELETE CASCADE) 연쇄 정리.
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def _make_asset(self, *, domain: str = "general") -> str:
        """최소 asset 1건 생성(node/edge 는 sync_graph_edges 가 만든다). 의료는 domain='medical'."""
        from src.registry.asset_persist import create_asset
        with self.db.transaction() as conn:
            aid = create_asset(
                conn, fs_path=f"/t/{uuid.uuid4().hex}.txt", modality="txt",
                domain=domain, file_hash=uuid.uuid4().hex)
        self._ids.append(aid)
        return str(aid)

    def _link_active(self, src_id: str, dst_id: str) -> None:
        """src→dst 로 유니크 topic 을 실은 **active** same_domain 엣지 1건 시딩.

        confidence=1.0 + auto_approve_min=1.0 → 자동승인(active). 유니크 topic 이라 세 함수
        조회가 이 픽스처 엣지만 집계한다.
        """
        from src.relations.graph_persist import sync_graph_edges
        edges = [{
            "target_media_item_id": dst_id, "relation_type_code": "same_domain",
            "topic_ko": self._topic, "topic_en": "e2e_topic",
            "subtopic_ko": self._subtopic, "subtopic_en": "baking",
            "confidence": 1.0, "reason": "056 e2e 주제 픽스처",
        }]
        up, _ = self.db.execute_in_transaction(
            lambda conn: sync_graph_edges(
                conn, source_asset_id=src_id, edges=edges,
                allowed_target_ids=frozenset({dst_id}), auto_approve_min=1.0),
            idempotent=False)
        self.assertEqual(up, 1)

    def _seed_graph(self):
        """A·B·C(일반)·M(의료) + 엣지 A—B, A—C, B—C, M—B(모두 active·유니크 topic)."""
        a = self._make_asset()
        b = self._make_asset()
        c = self._make_asset()
        m = self._make_asset(domain="medical")
        self._link_active(a, b)
        self._link_active(a, c)
        self._link_active(b, c)
        self._link_active(m, b)  # 의료 엣지 — 세 조회 모두에서 제외돼야 한다.
        return a, b, c, m

    def test_find_topic_neighbors_overlap_and_medical_excluded(self):
        from src.relations.topic_query import find_topic_neighbors
        a, b, c, m = self._seed_graph()

        out = self.db.execute_in_transaction(
            lambda conn: find_topic_neighbors(conn, asset_id=a, top_k=20),
            idempotent=True)

        by_id = {o["asset_id"]: o for o in out}
        # B·C 는 A 와 유니크 topic 공유(B—C 로 overlap 2). M 은 의료라 제외.
        self.assertIn(b, by_id)
        self.assertIn(c, by_id)
        self.assertNotIn(m, by_id)
        self.assertEqual(by_id[b]["overlap_weight"], 2)   # A—B, B—C
        self.assertEqual(by_id[c]["overlap_weight"], 2)   # A—C, B—C
        # B·C 는 A 의 직접 관계 이웃(A—B, A—C) → already_linked True
        self.assertTrue(by_id[b]["already_linked"])
        self.assertTrue(by_id[c]["already_linked"])
        self.assertEqual(by_id[b]["shared_topics"], [self._topic])
        # 조회행 계약: asset_id 는 str
        self.assertIsInstance(by_id[b]["asset_id"], str)

    def test_find_topic_neighbor_groups_pairs_and_medical_excluded(self):
        # 057 리뷰 권고 1: 같은주제 2단 중첩 그룹의 **실 SQL 의료 제외**(PHI)·쌍 매칭·already_linked 검증.
        # mock 단위(test_topic_query.py)는 _ACTIVE_MEDICAL_WHERE 가 실제로 M—B 를 거르는지 못 잡는다.
        from src.relations.topic_query import find_topic_neighbor_groups
        a, b, c, m = self._seed_graph()

        out = self.db.execute_in_transaction(
            lambda conn: find_topic_neighbor_groups(conn, asset_id=a),
            idempotent=True)

        # 유니크 topic 이라 대상 그룹은 1개(self._topic).
        groups = [g for g in out if g["topic_ko"] == self._topic]
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g["asset_count"], 2)   # {B,C} distinct — 대상 A 제외·M(의료) 제외

        # 하위주제 = self._subtopic 단일(대상 쌍 (topic, 제빵) 매칭)
        self.assertEqual([s["subtopic_ko"] for s in g["subtopics"]], [self._subtopic])
        sub = g["subtopics"][0]
        self.assertEqual(sub["asset_count"], 2)
        ids = {x["asset_id"] for x in sub["assets"]}
        self.assertEqual(ids, {b, c})
        # B·C 는 A 직접 이웃(A—B, A—C) → already_linked True
        self.assertTrue(all(x["already_linked"] for x in sub["assets"]))
        # 조회행 계약: asset_id 는 str
        self.assertTrue(all(isinstance(x["asset_id"], str) for x in sub["assets"]))

        # 핵심(PHI): 의료 자산 M 은 out 어디에도 없다(_ACTIVE_MEDICAL_WHERE 가 M—B 엣지 제외).
        #           대상 자신 A 도 제외.
        all_ids = {x["asset_id"] for gr in out for s in gr["subtopics"] for x in s["assets"]}
        self.assertNotIn(m, all_ids)
        self.assertNotIn(a, all_ids)

    def test_assets_in_topic_paging_and_medical_excluded(self):
        from src.relations.topic_query import assets_in_topic
        a, b, c, m = self._seed_graph()

        full = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, limit=50, offset=0),
            idempotent=True)
        ids = [r["asset_id"] for r in full["rows"]]
        self.assertEqual(full["total"], 3)          # {A,B,C} — M 제외
        self.assertEqual(set(ids), {a, b, c})
        self.assertNotIn(m, ids)
        self.assertEqual(ids, sorted(ids))          # asset_id asc 결정적

        # 페이징: limit=2 → 앞 2건, offset=2 → 나머지 1건. total 불변.
        p1 = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, limit=2, offset=0),
            idempotent=True)
        p2 = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, limit=2, offset=2),
            idempotent=True)
        self.assertEqual(len(p1["rows"]), 2)
        self.assertEqual(len(p2["rows"]), 1)
        self.assertEqual(p1["total"], 3)
        self.assertEqual(
            [r["asset_id"] for r in p1["rows"]] + [r["asset_id"] for r in p2["rows"]],
            sorted({a, b, c}))

        # subtopic 필터: 일치하면 3건, 없는 subtopic 이면 0건.
        hit = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, subtopic_ko=self._subtopic),
            idempotent=True)
        self.assertEqual(hit["total"], 3)
        miss = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, subtopic_ko="__없는것__"),
            idempotent=True)
        self.assertEqual(miss["total"], 0)

    def test_list_topics_asset_count_and_medical_excluded(self):
        from src.relations.topic_query import list_topics
        a, b, c, m = self._seed_graph()

        out = self.db.execute_in_transaction(
            lambda conn: list_topics(conn), idempotent=True)

        entries = [
            o for o in out
            if o["topic_ko"] == self._topic and o["subtopic_ko"] == self._subtopic
        ]
        self.assertEqual(len(entries), 1)           # 유니크 topic — 단일 엔트리
        self.assertEqual(entries[0]["asset_count"], 3)  # {A,B,C} distinct — M 제외


if __name__ == "__main__":
    unittest.main()
