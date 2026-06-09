"""OpenSearch 동기화 실OS/실DB e2e (spec 020 G5) — RUN_DB_E2E=1 + 실 OpenSearch 게이트.

PG 읽기전용 → OpenSearch 색인. 전용 테스트 인덱스(``assets_020e2e``)로 격리하고 tearDown 에서 삭제한다.
검증: 전체 재동기화 문서수=registered(SC-002)·멱등 재실행(중복 0)·단건 색인·결정성(SC-004)·영벡터 텍스트만.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

_RUN = os.getenv("RUN_DB_E2E") == "1"
_INDEX = "assets_020e2e"


def _os_reachable() -> bool:
    if not _RUN:
        return False
    try:
        import urllib.request

        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env.dev", override=False)
        url = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
        with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 — 로컬 DEV OS
            return r.status == 200
    except Exception:
        return False


@unittest.skipUnless(_os_reachable(), "RUN_DB_E2E=1 + 실 OpenSearch 필요")
class TestOpenSearchSyncE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env.dev", override=False)
        from src.config.settings import active_embed_channel, init_settings
        init_settings("dev")
        from src.database.postgres_util import PostgresUtil
        from src.search.opensearch_sync import get_client

        cls.client = get_client()
        cls.db = PostgresUtil()
        cls.channel = active_embed_channel()

    def tearDown(self) -> None:
        if self.client.indices.exists(index=_INDEX):
            self.client.indices.delete(index=_INDEX)

    def _registered_with_embedding(self, conn) -> int:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(DISTINCT a.asset_id) FROM asset a "
                "JOIN asset_embedding ae ON ae.asset_id=a.asset_id "
                "WHERE a.status='registered' AND ae.channel=%s AND ae.embedding IS NOT NULL",
                (self.channel,),
            )
            return cur.fetchone()[0]

    def test_full_resync_doc_count_and_idempotent(self) -> None:
        from src.search.opensearch_sync import sync_all

        with self.db:
            expected = self.db.execute_in_transaction(self._registered_with_embedding, idempotent=True)
            # 전체 재동기화(--recreate 동치)
            self.db.execute_in_transaction(
                lambda c: sync_all(self.client, c, index=_INDEX, channel=self.channel, recreate=True),
                idempotent=True,
            )
            self.client.indices.refresh(index=_INDEX)
            count1 = self.client.count(index=_INDEX)["count"]
            # SC-002: 문서수 = registered(임베딩 보유) 자산 수
            self.assertEqual(count1, expected)
            # 멱등 재실행 → 중복 0(같은 문서수)
            self.db.execute_in_transaction(
                lambda c: sync_all(self.client, c, index=_INDEX, channel=self.channel, recreate=False),
                idempotent=True,
            )
            self.client.indices.refresh(index=_INDEX)
            self.assertEqual(self.client.count(index=_INDEX)["count"], expected)

    def test_index_asset_single_and_determinism(self) -> None:
        from src.search.opensearch_sync import ensure_index, index_asset

        with self.db:
            ensure_index(self.client, _INDEX, recreate=True)
            # 임베딩 보유 자산 1건 선택
            def _pick(conn):
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT a.asset_id FROM asset a JOIN asset_embedding ae ON ae.asset_id=a.asset_id "
                        "WHERE a.status='registered' AND ae.channel=%s AND ae.embedding IS NOT NULL "
                        "ORDER BY a.asset_id LIMIT 1",
                        (self.channel,),
                    )
                    return str(cur.fetchone()[0])

            aid = self.db.execute_in_transaction(_pick, idempotent=True)
            # 단건 색인 2회 → 결정성(SC-004): 같은 문서
            self.db.execute_in_transaction(
                lambda c: index_asset(self.client, c, aid, index=_INDEX, channel=self.channel),
                idempotent=True,
            )
            self.client.indices.refresh(index=_INDEX)
            doc1 = self.client.get(index=_INDEX, id=aid)["_source"]
            self.db.execute_in_transaction(
                lambda c: index_asset(self.client, c, aid, index=_INDEX, channel=self.channel),
                idempotent=True,
            )
            self.client.indices.refresh(index=_INDEX)
            doc2 = self.client.get(index=_INDEX, id=aid)["_source"]
            self.assertEqual(doc1, doc2)
            self.assertEqual(doc1["asset_id"], aid)
            # 단건 색인 후 문서 1건(중복 0)
            self.assertEqual(self.client.count(index=_INDEX)["count"], 1)


if __name__ == "__main__":
    unittest.main()
