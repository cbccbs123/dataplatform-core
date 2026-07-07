"""백필 후 실DB SC 후조건 e2e(spec 058 G6 · T604 · SC-01/03/04/07).

RUN_DB_E2E 게이트(실 PostgreSQL 필요·기본 skip). ``scripts/backfill_topic_canonical.py --apply``
+ ``run_opensearch_resync`` 를 dev 에 적용한 **뒤** 상태를 단언한다(백필 후조건 회귀 가드). 사람이
``RUN_DB_E2E=1 python -m unittest tests.test_backfill_topic_canonical_e2e`` 로 재검증할 수 있다.
"""
from __future__ import annotations

import os
import unittest

_RUN_DB = os.environ.get("RUN_DB_E2E") == "1"

_MODALITY = ("텍스트", "오디오", "영상", "이미지", "text", "audio", "video", "image")

# 시드 병합 그룹(비정본 → 정본) — 백필 후 비정본 topic 은 0 이어야 한다(SC-01 동의어 수렴).
_MERGED = {
    "음식": "요리",
    "천문학": "천문",
    "자연재해": "재난",
    "관광": "여행",
    "산악": "등산",
    "가전": "전자제품",
    "수공예": "공예",
    "국방": "군사",
    "생물학": "생물",
}


def _distinct(cur, key: str) -> set[str]:
    cur.execute(
        f"SELECT DISTINCT topic->>'{key}' FROM graph_edge "
        f"WHERE status='active' AND COALESCE(topic->>'{key}','')<>''"
    )
    return {r[0] for r in cur.fetchall()}


@unittest.skipUnless(_RUN_DB, "실 DB e2e — RUN_DB_E2E=1 로만 실행")
class TestBackfillPostconditionsE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        from src.config.settings import init_settings

        load_dotenv(".env.dev", override=False)
        init_settings("dev")
        from src.database.postgres_util import PostgresUtil

        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.db.__exit__(None, None, None)

    def test_sc01_synonyms_converged(self):
        """SC-01: 병합 대상 비정본 topic 은 0 · 정본은 존재."""
        with self.db.connection() as conn, conn.cursor() as cur:
            for raw, cano in _MERGED.items():
                cur.execute(
                    "SELECT count(*) FROM graph_edge WHERE status='active' AND topic->>'topic_ko'=%s",
                    (raw,),
                )
                self.assertEqual(cur.fetchone()[0], 0, f"비정본 topic 잔존: {raw}")
                cur.execute(
                    "SELECT count(*) FROM graph_edge WHERE status='active' AND topic->>'topic_ko'=%s",
                    (cano,),
                )
                self.assertGreater(cur.fetchone()[0], 0, f"정본 topic 부재: {cano}")

    def test_sc03_no_topic_subtopic_overlap(self):
        """SC-03: topic 이자 subtopic 인 라벨 0(계층 일관)."""
        with self.db.connection() as conn, conn.cursor() as cur:
            overlap = _distinct(cur, "topic_ko") & _distinct(cur, "subtopic_ko")
        self.assertEqual(overlap, set(), f"계층 불일치 잔존: {sorted(overlap)}")

    def test_sc04_no_modality_subtopics(self):
        """SC-04: subtopic 의 매체어 0."""
        with self.db.connection() as conn, conn.cursor() as cur:
            subs = _distinct(cur, "subtopic_ko")
        mod = sorted(s for s in subs if s.lower() in _MODALITY)
        self.assertEqual(mod, [], f"모달리티 subtopic 잔존: {mod}")

    def test_sc07_distinct_topic_reduced(self):
        """SC-07: distinct topic 이 시드 정본 수(≤111)로 축소."""
        with self.db.connection() as conn, conn.cursor() as cur:
            topics = _distinct(cur, "topic_ko")
        self.assertLessEqual(len(topics), 111, f"distinct topic 미축소: {len(topics)}")


if __name__ == "__main__":
    unittest.main()
