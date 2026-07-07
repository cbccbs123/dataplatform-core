"""058 G1 — topic_registry·topic_alias 마이그레이션(v295) 실 DB 스키마 e2e.

무DB 환경에서는 자동 skip(다른 ``*_e2e`` 관례 일치). ``RUN_DB_E2E=1`` + 로컬
PostgreSQL(head=v295 적용)에서만 실행한다(사람/드라이버 게이트 — T101/T103).

검증 의도 (FR-101~103·SC-08 마이그레이션 가역·불변식)
    마이그레이션 적용 후 정본 레지스트리 2테이블의 **스키마 정합**을 실 DB 로 단언한다.
      - ``topic_registry``: topic_id(PK)·topic_ko(UNIQUE)·topic_en·embedding vector(1536)·source·created_at
      - ``topic_alias``   : raw_ko(PK)·canonical_ko(→topic_registry.topic_ko FK)·decided_by·created_at
      - embedding pgvector cosine 인덱스(repo 관례 hnsw·vector_cosine_ops)

선행: ``alembic -c alembic.ini upgrade head`` 로 v295 가 dev DB 에 반영돼 있어야 한다.
downgrade 가역(2테이블 drop·재 upgrade)은 apply 단계에서 별도 확인한다
(레포 관례: 마이그레이션 테스트는 alembic 을 직접 돌리지 않고 적용 후 상태를 단언).
"""
from __future__ import annotations

import dataclasses
import os
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


@unittest.skipUnless(_RUN, "실 DB 필요(RUN_DB_E2E=1)")
class TestTopicCanonicalizeMigrationV295(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        load_dotenv(_ENV, override=False)
        from src.config.settings import init_settings
        init_settings("dev")
        from src.database.postgres_util import PostgresUtil
        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.db.__exit__(None, None, None)

    def _columns(self, conn, table: str) -> dict[str, str]:
        """{column_name: udt_name} — udt_name 으로 vector 타입까지 확인."""
        rows = conn.execute(
            "SELECT column_name, udt_name FROM information_schema.columns "
            "WHERE table_name = %s",
            (table,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def _indexdefs(self, conn, table: str) -> list[str]:
        """테이블의 인덱스 정의 목록(indexdef) — 부분 유니크 인덱스의 WHERE 술어까지 포함."""
        rows = conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = %s", (table,)
        ).fetchall()
        return [r[0] for r in rows]

    def test_topic_registry_schema(self):
        with self.db.transaction() as conn:
            cols = self._columns(conn, "topic_registry")
            self.assertNotEqual(cols, {}, "topic_registry 테이블이 없음(v295 미적용?)")
            # v297: parent_topic 스코프 컬럼 추가(topic 층 NULL·subtopic 층 = 부모 topic_ko).
            for c in ("topic_id", "topic_ko", "topic_en", "embedding", "source",
                      "created_at", "parent_topic"):
                self.assertIn(c, cols, f"topic_registry.{c} 컬럼 누락")
            # embedding 은 pgvector vector 타입
            self.assertEqual(cols["embedding"], "vector", "embedding 이 vector 타입이 아님")

            # topic_id PK
            pk = conn.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'topic_registry'::regclass AND contype = 'p'"
            ).fetchone()
            self.assertIsNotNone(pk, "topic_registry PK 제약 없음")
            self.assertIn("topic_id", pk[0])

            # v297: topic_ko 단일 UNIQUE 제약은 드롭되고 **부모 스코프 부분 유니크 인덱스**로 대체됐다.
            defs = " ".join(self._indexdefs(conn, "topic_registry"))
            self.assertIn("UNIQUE", defs, "topic_registry 유니크 인덱스 없음")
            self.assertIn("topic_ko", defs, "topic_registry.topic_ko 유니크 인덱스 없음")
            self.assertIn("parent_topic IS NULL", defs, "topic 층 부분 유니크(parent NULL) 없음")

    def test_topic_registry_embedding_dim_1536(self):
        # embedding vector(1536) — 차원 헌법 불변식
        with self.db.transaction() as conn:
            dim = conn.execute(
                "SELECT a.atttypmod FROM pg_attribute a "
                "WHERE a.attrelid = 'topic_registry'::regclass AND a.attname = 'embedding'"
            ).fetchone()
            self.assertIsNotNone(dim)
            self.assertEqual(dim[0], 1536, "embedding 차원이 1536D 가 아님")

    def test_topic_registry_pgvector_cosine_index(self):
        with self.db.transaction() as conn:
            idx = conn.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename = 'topic_registry'"
            ).fetchall()
            defs = " ".join(d[0] for d in idx)
            self.assertIn("embedding", defs, "topic_registry.embedding 인덱스 없음")
            self.assertIn("vector_cosine_ops", defs, "pgvector cosine opclass 인덱스 없음")

    def test_topic_registry_scope_unique_indexes(self):
        # v297 FR-102v2: registry 부모 스코프 부분 유니크 인덱스 2개 —
        #   topic 층 = (topic_ko) WHERE parent_topic IS NULL,
        #   subtopic 층 = (parent_topic, topic_ko) WHERE parent_topic IS NOT NULL.
        with self.db.transaction() as conn:
            defs = self._indexdefs(conn, "topic_registry")
            root = [d for d in defs
                    if "UNIQUE" in d and "topic_ko" in d and "parent_topic IS NULL" in d]
            child = [d for d in defs
                     if "UNIQUE" in d and "parent_topic" in d and "topic_ko" in d
                     and "parent_topic IS NOT NULL" in d]
            self.assertTrue(root, "topic 층 부분 유니크 인덱스(parent NULL·topic_ko) 없음")
            self.assertTrue(child, "subtopic 층 부분 유니크 인덱스(parent, topic_ko) 없음")

    def test_topic_alias_schema(self):
        with self.db.transaction() as conn:
            cols = self._columns(conn, "topic_alias")
            self.assertNotEqual(cols, {}, "topic_alias 테이블이 없음(v295 미적용?)")
            # v297: parent_topic 스코프 컬럼 추가.
            for c in ("raw_ko", "canonical_ko", "decided_by", "created_at", "parent_topic"):
                self.assertIn(c, cols, f"topic_alias.{c} 컬럼 누락")

            # v297: raw_ko PK 는 드롭되고 **부모 스코프 부분 유니크 인덱스**로 대체됐다(PK 없음).
            pk = conn.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'topic_alias'::regclass AND contype = 'p'"
            ).fetchone()
            self.assertIsNone(pk, "topic_alias PK 는 v297 에서 부분 유니크 인덱스로 대체(드롭)돼야 함")
            defs = " ".join(self._indexdefs(conn, "topic_alias"))
            self.assertIn("UNIQUE", defs, "topic_alias 유니크 인덱스 없음")
            self.assertIn("raw_ko", defs, "topic_alias.raw_ko 유니크 인덱스 없음")

            # v297: canonical_ko FK 는 **완화(드롭)**됐다 — 부분 유니크 인덱스를 FK 대상으로 삼을 수
            #   없고 복합 FK 는 parent NULL(topic 층)에서 검사 스킵돼 무의미하므로, 앱 불변식으로 보증.
            fk = conn.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'topic_alias'::regclass AND contype = 'f'"
            ).fetchall()
            self.assertEqual(fk, [], "topic_alias FK 는 v297 에서 완화(드롭)돼야 함")

    def test_topic_alias_scope_unique_indexes(self):
        # v297 FR-102v2: alias 부모 스코프 부분 유니크 인덱스 2개(registry 와 동형).
        with self.db.transaction() as conn:
            defs = self._indexdefs(conn, "topic_alias")
            root = [d for d in defs
                    if "UNIQUE" in d and "raw_ko" in d and "parent_topic IS NULL" in d]
            child = [d for d in defs
                     if "UNIQUE" in d and "parent_topic" in d and "raw_ko" in d
                     and "parent_topic IS NOT NULL" in d]
            self.assertTrue(root, "topic 층 alias 부분 유니크 인덱스(parent NULL·raw_ko) 없음")
            self.assertTrue(child, "subtopic 층 alias 부분 유니크 인덱스(parent, raw_ko) 없음")

    def test_topic_registry_parent_scope_roundtrip(self):
        # v297: 같은 topic_ko 라도 부모가 다르면 공존(subtopic 층 스코프 유니크), 같은 (부모, topic_ko)
        #   재삽입은 유니크 위반. topic 층(parent NULL)은 topic_ko 단일 유니크.
        import psycopg
        from src.database.ids import uuid7
        sub = "e2e하위_" + os.urandom(4).hex()
        p1 = "e2e부모A_" + os.urandom(4).hex()
        p2 = "e2e부모B_" + os.urandom(4).hex()

        def _ins(conn, ko, parent):
            conn.execute(
                "INSERT INTO topic_registry (topic_id, topic_ko, source, parent_topic) "
                "VALUES (%s, %s, %s, %s)",
                (str(uuid7()), ko, "e2e", parent),
            )
        try:
            # 서로 다른 부모 아래 같은 subtopic 라벨 → 둘 다 성공(동음이의 보존)
            with self.db.transaction() as conn:
                _ins(conn, sub, p1)
                _ins(conn, sub, p2)
                n = conn.execute(
                    "SELECT count(*) FROM topic_registry WHERE topic_ko = %s", (sub,)
                ).fetchone()[0]
                self.assertEqual(n, 2)
            # 같은 (부모, topic_ko) 재삽입 → 부분 유니크 위반
            with self.assertRaises(psycopg.errors.UniqueViolation):
                with self.db.transaction() as conn:
                    _ins(conn, sub, p1)
        finally:
            with self.db.transaction() as conn:
                conn.execute(
                    "DELETE FROM topic_registry WHERE topic_ko = %s", (sub,)
                )


@unittest.skipUnless(_RUN, "실 DB 필요(RUN_DB_E2E=1)")
class TestTopicCanonicalizeWiringE2E(unittest.TestCase):
    """058 G8(T801) — 플래그-on 정본화 **배선** 실 DB e2e(생성시 동의어→정본 수렴 실증).

    무DB 환경에서는 자동 skip. ``RUN_DB_E2E=1`` + head=v295 dev DB 에서만 실행(사람/드라이버 게이트).

    검증 의도 (SC-01/03/04/05·FR-401·헌법 flag-off 동작 불변)
        단위(``tests/test_graph_persist.py``)는 mock 으로 배선 분기만 본다. 여기서는 **실 DB 라운드트립**으로
        생성시(플래그 on) 자유기입 동의어 라벨이 정본으로 수렴해 ``graph_edge.topic`` 에 저장되는지,
        모달리티 subtopic 이 비워지는지, 그리고 플래그 off 면 원본이 그대로 저장돼 동작이 불변인지 단언한다.

    자기완결(기존 시드 dev 상태 비의존)
        테스트가 유니크 라벨로 자기 registry(정본·임베딩 계산)·alias(동의어→정본) 를 시딩하고,
        생성한 asset/edge/registry/alias 를 tearDown 에서 정리한다. 운영 데이터(``graph_edge_topic_bak_058``
        등)·``.env`` 파일은 건드리지 않는다(플래그 토글은 settings 모듈 전역만 mock 으로 스와프).
    """

    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        load_dotenv(_ENV, override=False)
        from src.config.settings import init_settings
        init_settings("dev")
        from src.database.postgres_util import PostgresUtil
        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.db.__exit__(None, None, None)

    def setUp(self):
        # 충돌 방지 유니크 라벨 — 실 dev 레지스트리(정본 111·alias 120)와 겹치지 않아 단언이 견고.
        suffix = uuid.uuid4().hex[:8]
        self._canonical = "e2e정본_" + suffix
        self._canonical_en = "e2e_canon_" + suffix
        self._synonym = "e2e동의_" + suffix
        self._asset_ids: list = []           # create_asset → uuid.UUID(정리용)
        # 시드: 정본 register_topic(임베딩 계산·비-0노름 불변식) + 동의어 alias→정본 동결.
        from src.relations.topic_canonicalize import _freeze_alias, register_topic
        with self.db.transaction() as conn:
            register_topic(conn, self._canonical, self._canonical_en, source="e2e")
            _freeze_alias(conn, self._synonym, self._canonical, "e2e")

    def tearDown(self):
        # asset 삭제 → node(CASCADE) → graph_edge(CASCADE). 그다음 alias(FK)→registry 순서로 시드 정리.
        # 정본·동의어 둘 다 정리(동의어가 신규 정본으로 잘못 등록됐어도 회수·운영 데이터 불변).
        labels = [self._canonical, self._synonym]
        with self.db.transaction() as conn, conn.cursor() as cur:
            if self._asset_ids:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._asset_ids,))
            cur.execute("DELETE FROM topic_alias WHERE raw_ko = ANY(%s)", (labels,))
            cur.execute("DELETE FROM topic_registry WHERE topic_ko = ANY(%s)", (labels,))

    @contextmanager
    def _flag(self, enabled: bool):
        """``TOPIC_CANONICALIZE_ENABLED`` 를 **테스트 스코프에서만** 토글(.env 불변).

        settings 는 frozen dataclass 이므로 ``dataclasses.replace`` 로 플래그만 바꾼 사본을 만들고
        settings 모듈 전역 ``_SETTINGS`` 를 mock 으로 스와프한다 — ``graph_persist._topic_canonicalize_enabled``
        가 ``get_current_settings()`` 로 읽는 값을 블록 안에서만 바꾼다(종료 시 원복).
        """
        from src.config import settings as settings_mod
        base = settings_mod.get_current_settings()
        patched = dataclasses.replace(base, topic_canonicalize_enabled=enabled)
        with mock.patch.object(settings_mod, "_SETTINGS", patched):
            yield

    def _make_asset(self) -> str:
        from src.registry.asset_persist import create_asset
        with self.db.transaction() as conn:
            aid = create_asset(
                conn, fs_path=f"/t/{uuid.uuid4().hex}.txt", modality="txt",
                domain="general", file_hash=uuid.uuid4().hex)
        self._asset_ids.append(aid)
        return str(aid)

    def _persist_edge(self, src_id: str, dst_id: str, *, topic_ko: str, subtopic_ko: str) -> None:
        """src→dst same_domain active 엣지 1건 persist(현 배선 = 플래그 게이트 안에서 호출)."""
        from src.relations.graph_persist import sync_graph_edges
        edges = [{
            "target_media_item_id": dst_id, "relation_type_code": "same_domain",
            "topic_ko": topic_ko, "topic_en": "e2e_edge_en",
            "subtopic_ko": subtopic_ko, "subtopic_en": "text",
            "confidence": 1.0, "reason": "058 G8 배선 e2e",
        }]
        up, _ = self.db.execute_in_transaction(
            lambda conn: sync_graph_edges(
                conn, source_asset_id=src_id, edges=edges,
                allowed_target_ids=frozenset({dst_id}), auto_approve_min=1.0),
            idempotent=False)
        self.assertEqual(up, 1)

    def _read_edge_topic(self, src_id: str, dst_id: str) -> dict[str, str | None]:
        """(src,dst) asset 쌍의 graph_edge.topic 에서 topic_ko·subtopic_ko 회수(방향 무관·대칭 kind 대비)."""
        sql = """
            SELECT e.topic->>'topic_ko' AS topic_ko, e.topic->>'subtopic_ko' AS subtopic_ko
            FROM graph_edge e
            JOIN node n1 ON n1.node_id = e.src_node AND n1.node_kind = 'asset'
            JOIN node n2 ON n2.node_id = e.dst_node AND n2.node_kind = 'asset'
            WHERE (n1.asset_id = %s AND n2.asset_id = %s)
               OR (n1.asset_id = %s AND n2.asset_id = %s)
        """
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(sql, (src_id, dst_id, dst_id, src_id))
            row = cur.fetchone()
        self.assertIsNotNone(row, "graph_edge 미생성")
        return {"topic_ko": row[0], "subtopic_ko": row[1]}

    def test_flag_on_synonym_converges_and_modality_subtopic_emptied(self):
        """플래그 on: 동의어 topic_ko → 정본 수렴(alias 히트) · 모달리티 subtopic('텍스트') 비움(SC-01/04)."""
        src = self._make_asset()
        dst = self._make_asset()
        with self._flag(True):
            self._persist_edge(src, dst, topic_ko=self._synonym, subtopic_ko="텍스트")

        got = self._read_edge_topic(src, dst)
        self.assertEqual(got["topic_ko"], self._canonical)  # 동의어→정본 수렴(생성시 정규화)
        self.assertEqual(got["subtopic_ko"], "")            # 모달리티어 → 비움(계층·모달리티 규칙)

    def test_flag_off_keeps_raw_synonym_behavior_unchanged(self):
        """플래그 off(기본): 같은 동의어라도 원본 그대로 저장 · subtopic 원본 유지(동작 불변 재확인)."""
        src = self._make_asset()
        dst = self._make_asset()
        with self._flag(False):
            self._persist_edge(src, dst, topic_ko=self._synonym, subtopic_ko="텍스트")

        got = self._read_edge_topic(src, dst)
        self.assertEqual(got["topic_ko"], self._synonym)   # 정규화 안 됨(원본 동의어 유지)
        self.assertEqual(got["subtopic_ko"], "텍스트")      # 모달리티어도 비우지 않음(coerce 결과 그대로)

    def test_flag_on_deterministic_alias_hit_no_new_registry(self):
        """결정성(SC-05): 같은 동의어 2회 → 동일 정본(alias 히트) · 동의어는 신규 정본 등록 안 됨(LLM/등록 0)."""
        from src.relations.topic_canonicalize import lookup_alias
        resolved = []
        for _ in range(2):
            s = self._make_asset()
            d = self._make_asset()
            with self._flag(True):
                self._persist_edge(s, d, topic_ko=self._synonym, subtopic_ko="이미지")
            resolved.append(self._read_edge_topic(s, d)["topic_ko"])

        self.assertEqual(resolved[0], self._canonical)
        self.assertEqual(resolved[1], self._canonical)
        self.assertEqual(resolved[0], resolved[1])          # 두 번 다 동일 정본(결정적)

        with self.db.transaction() as conn:
            # 동의어는 정본 registry 에 새로 등록되지 않는다(alias 정확일치 경로만·kNN/judge/register 0).
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM topic_registry WHERE topic_ko = %s", (self._synonym,))
                self.assertEqual(cur.fetchone()[0], 0)
            # alias 는 시드 그대로 동의어→정본 유지(자기 정본으로 뒤집히지 않음).
            self.assertEqual(lookup_alias(conn, self._synonym), self._canonical)


if __name__ == "__main__":
    unittest.main()
