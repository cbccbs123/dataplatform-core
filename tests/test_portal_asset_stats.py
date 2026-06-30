import unittest
from datetime import datetime, timezone

from src.portal.asset_stats import asset_stats, query_assets


class _Cur:
    """execute 기록 + 미리 채운 결과를 순서대로 반환하는 fake 커서.

    COUNT(fetchone) → GROUP/SELECT(fetchall) 호출 순서대로 _results 를 소비한다.
    """
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


class AssetStatsShapeTest(unittest.TestCase):
    def test_stats_shape(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc).date()
        # COUNT → by_status → by_modality → by_domain → by_file_type → by_date 순으로 소비(6 쿼리)
        conn = _Conn([
            (10,),
            [("registered", 7), ("failed", 3)],
            [("text", 6), ("image", 4)],
            [("general", 9), ("unknown", 1)],
            [("pdf", 5), ("txt", 4), (None, 1)],
            [(ts, 10)],
        ])
        out = asset_stats(conn)
        self.assertEqual(out["total"], 10)
        self.assertEqual(out["by_status"][0], {"status": "registered", "count": 7})
        self.assertEqual(out["by_modality"][0], {"modality": "text", "count": 6})
        self.assertEqual(out["by_domain"][0], {"domain": "general", "count": 9})
        self.assertEqual(out["by_file_type"][0], {"file_type": "pdf", "count": 5})
        self.assertIsNone(out["by_file_type"][2]["file_type"])  # 확장자 없음(NULL)
        self.assertEqual(out["by_date"][0], {"date": ts.isoformat(), "count": 10})

    def test_excludes_medical_in_all_queries(self):
        conn = _Conn([(0,), [], [], [], [], []])
        asset_stats(conn)
        # 6개 SQL 모두 의료 제외 WHERE 절을 포함해야 함(total·status·modality·domain·file_type·date)
        self.assertEqual(len(conn._cur.calls), 6)
        for sql, _params in conn._cur.calls:
            self.assertIn("domain_label <> 'medical'", sql)

    def test_deterministic_order_sql(self):
        conn = _Conn([(0,), [], [], [], [], []])
        asset_stats(conn)
        self.assertIn("ORDER BY COUNT(*) DESC, status ASC", conn._cur.calls[1][0])
        self.assertIn("ORDER BY COUNT(*) DESC, modality ASC", conn._cur.calls[2][0])
        self.assertIn("ORDER BY COUNT(*) DESC, domain_label ASC", conn._cur.calls[3][0])
        self.assertIn("ORDER BY COUNT(*) DESC, ext ASC NULLS LAST", conn._cur.calls[4][0])  # file_type
        self.assertIn("GROUP BY d ORDER BY d ASC", conn._cur.calls[5][0])  # date 시간순


class QueryAssetsShapeTest(unittest.TestCase):
    def test_rows_shape_and_basename(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([
            (2,),
            [
                ("a1", "registered", "text", "general", "/data/raw/문서1.pdf", ts),
                ("a2", "failed", "image", "general", "/data/raw/사진.png", ts),
            ],
        ])
        out = query_assets(conn)
        self.assertEqual(out["total"], 2)
        self.assertEqual(out["rows"][0]["asset_id"], "a1")
        self.assertEqual(out["rows"][0]["status"], "registered")
        self.assertEqual(out["rows"][0]["modality"], "text")
        self.assertEqual(out["rows"][0]["domain_label"], "general")
        # file_name 은 fs_path 의 basename
        self.assertEqual(out["rows"][0]["file_name"], "문서1.pdf")
        self.assertEqual(out["rows"][1]["file_name"], "사진.png")
        self.assertEqual(out["rows"][0]["created_at"], ts.isoformat())

    def test_null_fs_path_file_name_none(self):
        conn = _Conn([(1,), [("a1", "registered", "text", "general", None, None)]])
        out = query_assets(conn)
        self.assertIsNone(out["rows"][0]["file_name"])
        self.assertIsNone(out["rows"][0]["created_at"])

    def test_always_excludes_medical(self):
        conn = _Conn([(0,), []])
        query_assets(conn)
        for sql, _params in conn._cur.calls:
            self.assertIn("domain_label <> 'medical'", sql)

    def test_filters_in_where_and_params(self):
        conn = _Conn([(0,), []])
        query_assets(conn, status="registered", modality="text", domain="general")
        count_sql, count_params = conn._cur.calls[0]
        select_sql, select_params = conn._cur.calls[1]
        # 필터는 %s 바인딩으로 WHERE 에 들어가고, 의료 제외는 항상 포함
        self.assertIn("status = %s", count_sql)
        self.assertIn("modality = %s", count_sql)
        self.assertIn("domain_label = %s", count_sql)
        self.assertIn("domain_label <> 'medical'", count_sql)
        self.assertEqual(count_params, ["registered", "text", "general"])
        # SELECT 도 동일 필터 + limit/offset 바인딩
        self.assertEqual(select_params, ["registered", "text", "general", 50, 0])

    def test_file_type_and_date_filters(self):
        dt = datetime(2026, 6, 1, tzinfo=timezone.utc)
        conn = _Conn([(0,), []])
        query_assets(conn, file_type="pdf", created_from=dt)
        count_sql, count_params = conn._cur.calls[0]
        # file_type 은 fs_path 확장자 식 = %s, 날짜는 created_at >= %s, 의료 제외 항상 포함
        self.assertIn("substring(fs_path from", count_sql)
        self.assertIn("created_at >= %s", count_sql)
        self.assertIn("domain_label <> 'medical'", count_sql)
        self.assertEqual(count_params, ["pdf", dt])

    def test_domain_medical_filter_contradiction_safe(self):
        # domain='medical' 요청 시 'domain_label<>medical' AND 'domain_label=medical' → 0행(PHI 안전)
        conn = _Conn([(0,), []])
        query_assets(conn, domain="medical")
        count_sql, _ = conn._cur.calls[0]
        self.assertIn("domain_label <> 'medical'", count_sql)
        self.assertIn("domain_label = %s", count_sql)

    def test_deterministic_order_sql(self):
        conn = _Conn([(0,), []])
        query_assets(conn)
        select_sql = conn._cur.calls[1][0]
        # created_at DESC + asset_id DESC tiebreak(결정적), 페이징 바인딩
        self.assertIn("ORDER BY created_at DESC, asset_id DESC LIMIT %s OFFSET %s", select_sql)

    def test_limit_offset_passthrough(self):
        conn = _Conn([(0,), []])
        query_assets(conn, limit=10, offset=20)
        _select_sql, select_params = conn._cur.calls[1]
        self.assertEqual(select_params, [10, 20])


if __name__ == "__main__":
    unittest.main()
