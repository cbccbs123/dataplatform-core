import unittest
from datetime import datetime, timezone

from src.portal.lineage_query import query_asset_lineage, query_lineage_feed


class _Cur:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): self.calls.append((sql, params))
    def fetchall(self): return self.rows


class _Conn:
    def __init__(self, rows): self._cur = _Cur(rows)
    def cursor(self): return self._cur


class _SeqCur:
    """execute 기록 + 미리 채운 결과를 순서대로 반환하는 fake 커서(COUNT→rows 2회 fetch)."""
    def __init__(self, results):
        self.calls = []
        self._results = list(results)
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params=None): self.calls.append((sql, params))
    def fetchone(self): return self._results.pop(0)
    def fetchall(self): return self._results.pop(0)


class _SeqConn:
    def __init__(self, results=()): self._cur = _SeqCur(results)
    def cursor(self): return self._cur


class QueryLineageTest(unittest.TestCase):
    def test_shape_and_order_query(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        rows = [("ingest.received.v1", "run_ingest", {}, {}, ts),
                ("ingest.registered.v1", "run_ingest", {}, {"channels": 2}, ts)]
        out = query_asset_lineage(_Conn(rows), "a1")
        self.assertEqual(out[0]["activity"], "ingest.received.v1")
        self.assertEqual(out[1]["generated"], {"channels": 2})
        # 시간순 정렬 SQL 사용 확인
        conn = _Conn(rows)
        query_asset_lineage(conn, "a1")
        self.assertIn("ORDER BY occurred_at ASC", conn._cur.calls[0][0])


class QueryLineageFeedTest(unittest.TestCase):
    def test_shape_and_total_no_filter(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        # feed 는 COUNT(*) 1행 → rows 목록, 2회 fetch
        conn = _SeqConn([(2,), [("l1", "a1", "ingest.received.v1", "run_ingest", ts),
                                ("l2", "a9", "ingest.registered.v1", "run_ingest", ts)]])
        out = query_lineage_feed(conn)
        self.assertEqual(out["total"], 2)
        # rows 에 asset_id 포함(전 자산 피드)
        self.assertEqual(out["rows"][0]["asset_id"], "a1")
        self.assertEqual(out["rows"][1]["activity"], "ingest.registered.v1")
        self.assertEqual(out["rows"][0]["occurred_at"], ts.isoformat())

    def test_order_by_occurred_at_desc_sql(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _SeqConn([(1,), [("l1", "a1", "ingest.received.v1", "run_ingest", ts)]])
        query_lineage_feed(conn)
        # 두 번째 execute(rows 조회)에 시간역순·tiebreak 정렬 SQL 사용 확인
        rows_sql = conn._cur.calls[1][0]
        self.assertIn("ORDER BY occurred_at DESC, lineage_id DESC", rows_sql)

    def test_activity_filter_in_where(self):
        conn = _SeqConn([(0,), []])
        query_lineage_feed(conn, activity="ingest.received.v1")
        # COUNT·rows 두 execute 모두 WHERE 에 activity 조건 + 바인딩 파라미터
        count_sql, count_params = conn._cur.calls[0]
        self.assertIn("activity = %s", count_sql)
        self.assertIn("ingest.received.v1", count_params)


if __name__ == "__main__":
    unittest.main()
