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

import os
import unittest
from pathlib import Path

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

    def test_topic_registry_schema(self):
        with self.db.transaction() as conn:
            cols = self._columns(conn, "topic_registry")
            self.assertNotEqual(cols, {}, "topic_registry 테이블이 없음(v295 미적용?)")
            for c in ("topic_id", "topic_ko", "topic_en", "embedding", "source", "created_at"):
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

            # topic_ko UNIQUE
            uniq = conn.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'topic_registry'::regclass AND contype = 'u'"
            ).fetchall()
            self.assertTrue(
                any("topic_ko" in u[0] for u in uniq),
                "topic_registry.topic_ko UNIQUE 제약 없음",
            )

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

    def test_topic_alias_schema(self):
        with self.db.transaction() as conn:
            cols = self._columns(conn, "topic_alias")
            self.assertNotEqual(cols, {}, "topic_alias 테이블이 없음(v295 미적용?)")
            for c in ("raw_ko", "canonical_ko", "decided_by", "created_at"):
                self.assertIn(c, cols, f"topic_alias.{c} 컬럼 누락")

            # raw_ko PK
            pk = conn.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'topic_alias'::regclass AND contype = 'p'"
            ).fetchone()
            self.assertIsNotNone(pk, "topic_alias PK 제약 없음")
            self.assertIn("raw_ko", pk[0])

            # canonical_ko FK → topic_registry(topic_ko)
            fk = conn.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conrelid = 'topic_alias'::regclass AND contype = 'f'"
            ).fetchall()
            self.assertTrue(fk, "topic_alias FK 제약 없음")
            joined = " ".join(f[0] for f in fk)
            self.assertIn("canonical_ko", joined)
            self.assertIn("topic_registry", joined)
            self.assertIn("topic_ko", joined)

    def test_topic_alias_fk_roundtrip(self):
        # FK 무결성: 미등록 canonical 은 거부, 등록 후 alias 삽입 성공(가역 검증 겸)
        import psycopg
        topic_ko = "e2e정본_v295_" + os.urandom(4).hex()
        raw_ko = "e2e원본_v295_" + os.urandom(4).hex()
        try:
            # 미등록 canonical → FK 위반
            with self.assertRaises(psycopg.errors.ForeignKeyViolation):
                with self.db.transaction() as conn:
                    conn.execute(
                        "INSERT INTO topic_alias (raw_ko, canonical_ko, decided_by) "
                        "VALUES (%s, %s, %s)",
                        (raw_ko, topic_ko, "e2e"),
                    )
            # 정본 등록 후 alias 삽입 성공
            from src.database.ids import uuid7
            with self.db.transaction() as conn:
                conn.execute(
                    "INSERT INTO topic_registry (topic_id, topic_ko, source) "
                    "VALUES (%s, %s, %s)",
                    (str(uuid7()), topic_ko, "e2e"),
                )
                conn.execute(
                    "INSERT INTO topic_alias (raw_ko, canonical_ko, decided_by) "
                    "VALUES (%s, %s, %s)",
                    (raw_ko, topic_ko, "e2e"),
                )
                got = conn.execute(
                    "SELECT canonical_ko FROM topic_alias WHERE raw_ko = %s", (raw_ko,)
                ).fetchone()
                self.assertEqual(got[0], topic_ko)
        finally:
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM topic_alias WHERE raw_ko = %s", (raw_ko,))
                conn.execute("DELETE FROM topic_registry WHERE topic_ko = %s", (topic_ko,))


if __name__ == "__main__":
    unittest.main()
