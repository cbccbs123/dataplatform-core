"""006 검색 서비스 진입점 e2e — search_hybrid 전경로를 실 DB로 검증(T016).

structured 를 주입해 LLM 질의 구조화(structure_user_query)를 건너뛴다(="LLM 주입" 취지).
질의 임베딩은 실 ST/CLIP 모델을 사용하므로 RUN_DB_E2E + 모델 가용 환경에서만 동작한다.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from dotenv import load_dotenv

from src.search.search_service import search_hybrid

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만(실 DB + 모델)")
class TestSearchServiceE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(_ENV, override=False)
        from src.config.settings import init_settings

        init_settings("dev")  # 임베딩 모델 설정(text_embedding_model 등) 활성화
        from src.database.postgres_util import PostgresUtil

        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db.__exit__(None, None, None)

    def setUp(self) -> None:
        self._ids: list = []

    def tearDown(self) -> None:
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def test_search_hybrid_returns_buckets_with_loaded_asset(self) -> None:
        from tests.test_media_search_rewire import _make_search_asset, _unit_vec

        token = "유일토큰qwerty 워크숍 발표 자료"
        aid = _make_search_asset(
            self.db, self._ids, fts=token, summary="작년 워크숍 발표", st_vector=_unit_vec(0), modality="txt"
        )
        structured = {
            "semantic_query": token,
            "semantic_query_en": "workshop presentation",
            "keywords": [],
            "keywords_en": [],
            "date_start": None,
            "date_end": None,
        }
        out = search_hybrid(token, structured=structured)

        # 응답 모양: 전체 버킷 + meta(structured 그대로)
        self.assertEqual(
            set(out["results"].keys()), {"text_documents", "audio", "image", "video"}
        )
        self.assertEqual(out["meta"]["structured"]["semantic_query"], token)
        # 적재한 텍스트 자산이 FTS 매칭으로 text_documents 버킷에 떠야(레거시 참조 없이 실 DB)
        ids = [r["id"] for r in out["results"]["text_documents"]]
        self.assertIn(aid, ids)


if __name__ == "__main__":
    unittest.main()
