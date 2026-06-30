import unittest
from datetime import datetime, timezone

from src.portal.access_log import (
    access_log_stats,
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


if __name__ == "__main__":
    unittest.main()
