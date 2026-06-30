import unittest
from datetime import datetime, timezone

from src.portal.lineage_query import query_asset_lineage


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


if __name__ == "__main__":
    unittest.main()
