import unittest
from datetime import datetime, timezone

from src.portal.lineage_query import (
    lineage_stats,
    lineage_timeline,
    query_asset_lineage,
    query_lineage_feed,
)


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
        sql = conn._cur.calls[0][0]
        self.assertIn("ORDER BY al.occurred_at ASC", sql)
        self.assertIn("a.domain_label <> 'medical'", sql)  # 의료 제외 조인(헌법 10조)


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
        self.assertIn("ORDER BY al.occurred_at DESC, al.lineage_id DESC", rows_sql)
        self.assertIn("a.domain_label <> 'medical'", rows_sql)  # 의료 제외(헌법 10조)

    def test_activity_filter_in_where(self):
        conn = _SeqConn([(0,), []])
        query_lineage_feed(conn, activity="ingest.received.v1")
        # COUNT·rows 두 execute 모두 WHERE 에 activity 조건 + 바인딩 파라미터
        count_sql, count_params = conn._cur.calls[0]
        self.assertIn("al.activity = %s", count_sql)
        self.assertIn("ingest.received.v1", count_params)

    def test_asset_dimension_filters(self):
        # 자산 차원 필터(modality·status·file_ext)는 asset 조인(a)으로 WHERE 에 들어간다(대시보드 슬라이스).
        conn = _SeqConn([(0,), []])
        query_lineage_feed(conn, modality="video", status="registered", file_ext="mp4")
        count_sql, count_params = conn._cur.calls[0]
        self.assertIn("a.modality = %s", count_sql)
        self.assertIn("a.status = %s", count_sql)
        self.assertIn("substring(a.fs_path from", count_sql)  # file_ext=확장자
        self.assertIn("a.domain_label <> 'medical'", count_sql)  # 의료 제외 유지
        for v in ("video", "registered", "mp4"):
            self.assertIn(v, count_params)


class LineageStatsTest(unittest.TestCase):
    def test_shape_and_medical_excluded(self):
        d = datetime(2026, 6, 30, tzinfo=timezone.utc).date()
        # COUNT → by_activity → by_day → by_modality → by_status → by_file_ext (6 쿼리)
        conn = _SeqConn([
            (12,),
            [("ingest.registered.v1", 8), ("relations.proposed.v1", 4)],
            [(d, 12)],
            [("text", 7), ("video", 5)],
            [("registered", 10), ("failed", 2)],
            [("txt", 6), ("mp4", 5), (None, 1)],
        ])
        out = lineage_stats(conn)
        self.assertEqual(out["total"], 12)
        self.assertEqual(out["by_activity"][0], {"activity": "ingest.registered.v1", "count": 8})
        self.assertEqual(out["by_day"][0], {"day": d.isoformat(), "count": 12})
        self.assertEqual(out["by_modality"][0]["modality"], "text")
        self.assertEqual(out["by_status"][0]["status"], "registered")
        self.assertEqual(out["by_file_ext"][0]["file_ext"], "txt")
        # 6개 SQL 모두 의료 제외 조인 포함
        self.assertEqual(len(conn._cur.calls), 6)
        for sql, _p in conn._cur.calls:
            self.assertIn("a.domain_label <> 'medical'", sql)


class LineageTimelineTest(unittest.TestCase):
    def test_single_series_no_group_by(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        out = lineage_timeline(_Conn([(ts, 5)]), interval="day", group_by=None)
        self.assertEqual(out["interval"], "day")
        self.assertEqual(out["buckets"][0]["count"], 5)
        self.assertNotIn("series", out)

    def test_multi_series_group_by_activity(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        # 그룹 행: (key, bucket, count) — key ASC·bucket ASC 정렬 가정
        conn = _Conn([
            ("ingest.received.v1", ts, 3),
            ("ingest.received.v1", ts, 2),
            ("ingest.registered.v1", ts, 4),
        ])
        out = lineage_timeline(conn, group_by="activity")
        self.assertEqual(out["group_by"], "activity")
        self.assertEqual(len(out["series"]), 2)  # 2개 활동 시리즈로 피벗
        self.assertEqual(out["series"][0]["key"], "ingest.received.v1")
        self.assertEqual(len(out["series"][0]["buckets"]), 2)
        self.assertEqual(out["series"][1]["key"], "ingest.registered.v1")
        # group_by 컬럼은 화이트리스트 매핑이라 SQL 에 al.activity 로 들어감(인젝션 안전)
        self.assertIn("al.activity AS key", conn._cur.calls[0][0])
        self.assertIn("a.domain_label <> 'medical'", conn._cur.calls[0][0])


if __name__ == "__main__":
    unittest.main()
