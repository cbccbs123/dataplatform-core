"""013 슬라이스 실 DB e2e — 기록→조회·통계·계보 (RUN_DB_E2E=1 에서만).

미설정(기본) 시 skip — 회귀 suite 0 영향. 실 DB(.env.dev) 필요.
access_log 에 고유 marker user_id 로 1행 적재 후 조회·집계 재현을 검증하고(SC-005/006/006a),
registered 자산의 계보 타임라인이 비어있지 않음을 확인한다(SC-004).
"""
import os
import unittest
from pathlib import Path

from src.database.ids import uuid7


@unittest.skipUnless(os.environ.get("RUN_DB_E2E") == "1", "RUN_DB_E2E=1 에서만(실 DB)")
class HistoryE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env.dev", override=False)
        from src.config.settings import init_settings
        init_settings("dev")

    def test_record_then_query_and_stats(self):
        from src.database.postgres_util import PostgresUtil
        from src.portal.access_log import access_log_stats, query_access_logs, record_access

        marker = f"e2e-{uuid7()}"  # 격리용 고유 user_id
        db = PostgresUtil()
        with db:
            db.execute_in_transaction(
                lambda c: record_access(c, action="search", user_id=marker), idempotent=False)
            out = db.execute_in_transaction(
                lambda c: query_access_logs(c, user_id=marker), idempotent=True)
            self.assertEqual(out["total"], 1)
            self.assertEqual(out["rows"][0]["action"], "search")
            self.assertEqual(out["rows"][0]["user_id"], marker)
            stats = db.execute_in_transaction(
                lambda c: access_log_stats(c), idempotent=True)
            self.assertGreaterEqual(stats["total"], 1)
            self.assertTrue(any(r["action"] == "search" for r in stats["by_action"]))

    def test_lineage_of_registered_asset(self):
        from src.database.postgres_util import PostgresUtil
        from src.portal.lineage_query import query_asset_lineage

        db = PostgresUtil()
        with db:
            aid = db.execute_in_transaction(_first_registered_asset, idempotent=True)
            if aid is None:
                self.skipTest("registered 자산 없음")
            acts = db.execute_in_transaction(
                lambda c: query_asset_lineage(c, aid), idempotent=True)
            self.assertTrue(acts)  # 수집 활동 타임라인 존재
            self.assertTrue(all("activity" in a and "occurred_at" in a for a in acts))

    def test_lineage_feed_and_access_timeline(self):
        from src.database.postgres_util import PostgresUtil
        from src.portal.access_log import access_log_timeline
        from src.portal.lineage_query import query_lineage_feed

        db = PostgresUtil()
        with db:
            feed = db.execute_in_transaction(
                lambda c: query_lineage_feed(c, limit=5), idempotent=True)
            self.assertIn("rows", feed)
            self.assertIn("total", feed)
            self.assertLessEqual(len(feed["rows"]), 5)  # 페이징 limit 준수
            self.assertTrue(all("asset_id" in r for r in feed["rows"]))
            tl = db.execute_in_transaction(
                lambda c: access_log_timeline(c, interval="day"), idempotent=True)
            self.assertEqual(tl["interval"], "day")
            self.assertTrue(all("bucket" in b and "count" in b for b in tl["buckets"]))

    def test_lineage_stats_and_timeline_multiseries(self):
        from src.database.postgres_util import PostgresUtil
        from src.portal.access_log import access_log_timeline
        from src.portal.lineage_query import lineage_stats, lineage_timeline

        db = PostgresUtil()
        with db:
            st = db.execute_in_transaction(lineage_stats, idempotent=True)
            for k in ("total", "by_activity", "by_day", "by_modality", "by_status", "by_file_ext"):
                self.assertIn(k, st)
            self.assertEqual(sum(a["count"] for a in st["by_activity"]), st["total"])  # 활동 합=총계
            # 계보 타임라인 멀티시리즈(group_by=activity)
            tl = db.execute_in_transaction(
                lambda c: lineage_timeline(c, group_by="activity"), idempotent=True)
            self.assertEqual(tl["group_by"], "activity")
            self.assertTrue(all("key" in s and "buckets" in s for s in tl["series"]))
            # access 타임라인 멀티시리즈(group_by=action)
            at = db.execute_in_transaction(
                lambda c: access_log_timeline(c, group_by="action"), idempotent=True)
            self.assertEqual(at["group_by"], "action")
            self.assertTrue(all("key" in s for s in at["series"]))

    def test_asset_stats_and_list(self):
        from src.database.postgres_util import PostgresUtil
        from src.portal.asset_stats import asset_stats, query_assets

        db = PostgresUtil()
        with db:
            stats = db.execute_in_transaction(asset_stats, idempotent=True)
            self.assertIn("by_status", stats)
            self.assertGreaterEqual(stats["total"], 0)
            # FSM 단계 분포 합 == 총계(의료 제외 일관)
            self.assertEqual(sum(s["count"] for s in stats["by_status"]), stats["total"])
            lst = db.execute_in_transaction(
                lambda c: query_assets(c, limit=5), idempotent=True)
            self.assertLessEqual(len(lst["rows"]), 5)
            self.assertTrue(all("status" in r and "file_name" in r for r in lst["rows"]))


def _first_registered_asset(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT asset_id FROM asset WHERE status='registered' ORDER BY asset_id LIMIT 1")
        row = cur.fetchone()
    return str(row[0]) if row else None


if __name__ == "__main__":
    unittest.main()
