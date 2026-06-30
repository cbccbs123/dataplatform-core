import unittest
from datetime import datetime, timezone

from src.portal.access_log import (
    access_log_stats,
    access_log_timeline,
    derive_access_action,
    query_access_logs,
    record_access,
)


class _Cur:
    """execute 기록 + 미리 채운 결과를 순서대로 반환하는 fake 커서."""
    def __init__(self, results):
        self.calls = []
        self._results = list(results)
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): self.calls.append((sql, params))
    def fetchone(self): return self._results.pop(0)
    def fetchall(self): return self._results.pop(0)


class _Conn:
    def __init__(self, results=()): self._cur = _Cur(results)
    def cursor(self): return self._cur


class DeriveActionTest(unittest.TestCase):
    def test_routes(self):
        self.assertEqual(derive_access_action("GET", "/search"), ("search", None))
        self.assertEqual(derive_access_action("GET", "/assets/abc"), ("asset_view", "abc"))
        self.assertEqual(derive_access_action("GET", "/assets/abc/download"), ("download", "abc"))
        self.assertEqual(derive_access_action("GET", "/assets/abc/bundle"), ("bundle", "abc"))

    def test_non_data_routes_none(self):
        for p in ("/health", "/me", "/auth/token", "/access-logs", "/access-logs/stats",
                  "/assets/abc/lineage", "/assets/"):
            self.assertIsNone(derive_access_action("GET", p), p)
        self.assertIsNone(derive_access_action("POST", "/search"))  # 비 GET


class RecordAccessTest(unittest.TestCase):
    def test_inserts_one_row_with_uuid(self):
        conn = _Conn()
        aid = record_access(conn, action="search", user_id="u1")
        self.assertTrue(aid)  # access_id 반환
        self.assertEqual(len(conn._cur.calls), 1)
        sql, params = conn._cur.calls[0]
        self.assertIn("INSERT INTO access_log", sql)
        self.assertEqual(params[2], "u1")     # user_id
        self.assertEqual(params[3], "search")  # action


class QueryStatsShapeTest(unittest.TestCase):
    def test_query_shape(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([(2,), [("id1", "search", "u1", None, ts), ("id2", "asset_view", "u1", "a9", ts)]])
        out = query_access_logs(conn, user_id="u1")
        self.assertEqual(out["total"], 2)
        self.assertEqual(out["rows"][0]["action"], "search")
        self.assertEqual(out["rows"][1]["asset_id"], "a9")

    def test_stats_shape(self):
        conn = _Conn([(3,), [("search", 2), ("asset_view", 1)], [("u1", 3)]])
        out = access_log_stats(conn)
        self.assertEqual(out["total"], 3)
        self.assertEqual(out["by_action"][0], {"action": "search", "count": 2})
        self.assertEqual(out["by_user"][0], {"user_id": "u1", "count": 3})


class TimelineShapeTest(unittest.TestCase):
    def test_bucket_shape(self):
        b0 = datetime(2026, 6, 29, tzinfo=timezone.utc)
        b1 = datetime(2026, 6, 30, tzinfo=timezone.utc)
        # timeline 은 GROUP BY 단일 execute → fetchall 1회
        conn = _Conn([[(b0, 2), (b1, 5)]])
        out = access_log_timeline(conn)
        self.assertEqual(out["interval"], "day")
        self.assertEqual(out["buckets"][0], {"bucket": b0.isoformat(), "count": 2})
        self.assertEqual(out["buckets"][1], {"bucket": b1.isoformat(), "count": 5})

    def test_interval_whitelist_fallback(self):
        # 화이트리스트 밖 interval("year")은 day 로 폴백(인젝션 방지)
        conn = _Conn([[]])
        out = access_log_timeline(conn, interval="year")
        self.assertEqual(out["interval"], "day")
        sql = conn._cur.calls[0][0]
        self.assertIn("date_trunc('day'", sql)
        self.assertNotIn("year", sql)

    def test_interval_hour_passthrough(self):
        # 화이트리스트 안 interval("hour")은 그대로 사용
        conn = _Conn([[]])
        out = access_log_timeline(conn, interval="hour")
        self.assertEqual(out["interval"], "hour")
        self.assertIn("date_trunc('hour'", conn._cur.calls[0][0])

    def test_action_filter_in_where(self):
        conn = _Conn([[]])
        access_log_timeline(conn, action="search")
        sql, params = conn._cur.calls[0]
        self.assertIn("action = %s", sql)
        self.assertIn("search", params)


if __name__ == "__main__":
    unittest.main()
