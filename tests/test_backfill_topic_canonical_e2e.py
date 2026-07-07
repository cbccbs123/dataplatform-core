"""백필 v2 후 실DB SC 후조건 e2e (spec 058 v2 · G12 · T1202 · SC-01v2/02v2/07v2).

RUN_DB_E2E 게이트(실 PostgreSQL 필요·기본 skip). 아래 순서를 dev 에 적용한 **뒤** 상태를 단언한다:
  1) ``scripts/seed_topic_registry.py --apply``  (taxonomy 28[미분류 포함] + alias 선시드 117·§3 매핑)
  2) ``scripts/backfill_topic_canonical.py --apply``  (쌍 단위 재작성·백업 graph_edge_topic_bak_058_v2)
  3) ``run_opensearch_resync --env dev``  (topics/subtopics 재투영·파리티)
사람이 ``RUN_DB_E2E=1 python -m unittest tests.test_backfill_topic_canonical_e2e`` 로 재검증한다.

v2 개정(2026-07-07·닫힌 분류체계 전환): v1 은 자유기입 동의어를 정본으로 **병합**했으나(음식→요리),
v2 는 topic 을 **닫힌 27+미분류**로 **분류**하고 옛 topic 은 대개 subtopic 으로 내려간다(요리→음식·요리,
등산→스포츠·레저>등산). 따라서 검증도 "병합 정본 존재" 대신 **닫힌 집합 소속·계층 일관·단일 부모**로 바뀐다.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

_RUN_DB = os.environ.get("RUN_DB_E2E") == "1"

_MODALITY = ("텍스트", "오디오", "영상", "이미지", "text", "audio", "video", "image")

_SEED_PATH = (
    Path(__file__).resolve().parents[1]
    / "specs"
    / "058-relation-topic-canonicalization"
    / "taxonomy_seed.json"
)

# v2 백필 후 topic 으로 남으면 안 되는 옛 자유기입 라벨(닫힌 집합 밖·subtopic 으로 내려가거나 분류됨).
# 요리→음식·요리, 등산→스포츠·레저, 천문학→과학, 에너지→경제·산업(§3 alias 선시드) 등.
_OLD_FREEFORM_TOPICS = ["요리", "음식", "등산", "천문학", "천문", "에너지", "반도체", "기타"]


def _distinct(cur, key: str) -> set[str]:
    cur.execute(
        f"SELECT DISTINCT topic->>'{key}' FROM graph_edge "
        f"WHERE status='active' AND COALESCE(topic->>'{key}','')<>''"
    )
    return {r[0] for r in cur.fetchall()}


def _closed_topic_kos() -> set[str]:
    with open(_SEED_PATH, encoding="utf-8") as f:
        return {str(t["topic_ko"]) for t in json.load(f)["topics"]}


@unittest.skipUnless(_RUN_DB, "실 DB e2e — RUN_DB_E2E=1 로만 실행")
class TestBackfillV2PostconditionsE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        from src.config.settings import init_settings

        load_dotenv(".env.dev", override=False)
        init_settings("dev")
        from src.database.postgres_util import PostgresUtil

        cls.db = PostgresUtil()
        cls.db.__enter__()
        cls.closed = _closed_topic_kos()

    @classmethod
    def tearDownClass(cls):
        cls.db.__exit__(None, None, None)

    def test_sc07_topics_within_closed_taxonomy(self):
        """SC-07v2: distinct topic 이 전부 닫힌 27+미분류 안에 있고(신규 topic 0) 수는 28 이하."""
        with self.db.connection() as conn, conn.cursor() as cur:
            topics = _distinct(cur, "topic_ko")
        off_list = sorted(topics - self.closed)
        self.assertEqual(off_list, [], f"닫힌 집합 밖 topic 잔존: {off_list}")
        self.assertLessEqual(len(topics), 28, f"distinct topic 초과: {len(topics)}")

    def test_sc07_old_freeform_topics_gone(self):
        """SC-07v2: 옛 자유기입 topic(요리·등산·천문학·에너지·기타 등)은 topic 층에서 0."""
        with self.db.connection() as conn, conn.cursor() as cur:
            for raw in _OLD_FREEFORM_TOPICS:
                cur.execute(
                    "SELECT count(*) FROM graph_edge WHERE status='active' AND topic->>'topic_ko'=%s",
                    (raw,),
                )
                self.assertEqual(cur.fetchone()[0], 0, f"옛 topic 잔존: {raw}")

    def test_sc01_no_topic_subtopic_overlap(self):
        """SC-01v2: topic 이자 subtopic 인 라벨 0(계층 일관·미분류 개명으로 기타 동음이의 해소)."""
        with self.db.connection() as conn, conn.cursor() as cur:
            overlap = _distinct(cur, "topic_ko") & _distinct(cur, "subtopic_ko")
        self.assertEqual(overlap, set(), f"계층 불일치 잔존: {sorted(overlap)}")

    def test_sc01_no_modality_subtopics(self):
        """SC-01v2: subtopic 의 매체어 0."""
        with self.db.connection() as conn, conn.cursor() as cur:
            subs = _distinct(cur, "subtopic_ko")
        mod = sorted(s for s in subs if s.lower() in _MODALITY)
        self.assertEqual(mod, [], f"모달리티 subtopic 잔존: {mod}")

    def test_sc01_gimbap_single_parent_food(self):
        """SC-01v2 핵심: 김밥 subtopic 의 부모 topic 은 음식·요리 **단독**(다중 부모 구조적 불가)."""
        with self.db.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT topic->>'topic_ko' FROM graph_edge "
                "WHERE status='active' AND topic->>'subtopic_ko'='김밥'"
            )
            parents = {r[0] for r in cur.fetchall()}
        self.assertEqual(parents, {"음식·요리"}, f"김밥 다중/오부모: {sorted(parents)}")

    def test_sc01_guitar_subtopic_preserved(self):
        """SC-01v2 동음이의: catch-all 을 미분류로 개명해 '음악>기타'(guitar) subtopic 이 보존된다."""
        with self.db.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM graph_edge WHERE status='active' "
                "AND topic->>'topic_ko'='음악' AND topic->>'subtopic_ko'='기타'"
            )
            self.assertGreater(cur.fetchone()[0], 0, "guitar subtopic '음악>기타' 소실")

    def test_sc02_unclassified_rate_low(self):
        """SC-02v2: 미분류(catch-all) 비율이 낮다(§3 커버리지 완전 → 사실상 0%·상한 5%)."""
        with self.db.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM graph_edge WHERE status='active'")
            total = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM graph_edge WHERE status='active' AND topic->>'topic_ko'='미분류'"
            )
            n_unc = cur.fetchone()[0]
        self.assertGreater(total, 0)
        self.assertLessEqual(n_unc / total, 0.05, f"미분류율 과다: {n_unc}/{total}")

    def test_energy_reclassified_to_economy(self):
        """반영2(§3 alias 선시드): 에너지 계열은 경제·산업으로 분류(과거 LLM 과학 오분류 방지)."""
        with self.db.connection() as conn, conn.cursor() as cur:
            # 에너지 는 topic 으로 남지 않는다(경제·산업으로 분류·subtopic 화).
            cur.execute(
                "SELECT count(*) FROM graph_edge WHERE status='active' AND topic->>'topic_ko'='에너지'"
            )
            self.assertEqual(cur.fetchone()[0], 0, "에너지 가 topic 으로 잔존")
            # 태양광 subtopic 은 경제·산업 아래(에너지 alias 효과의 대표 케이스).
            cur.execute(
                "SELECT DISTINCT topic->>'topic_ko' FROM graph_edge "
                "WHERE status='active' AND topic->>'subtopic_ko'='태양광'"
            )
            parents = {r[0] for r in cur.fetchall()}
        self.assertIn("경제·산업", parents, f"태양광 부모에 경제·산업 없음: {sorted(parents)}")


if __name__ == "__main__":
    unittest.main()
