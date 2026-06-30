import unittest
from datetime import datetime, timezone

from src.portal.asset_stats import (
    asset_stats,
    asset_timeline,
    modality_detail,
    query_assets,
)


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
        # COUNT → by_status → by_modality → by_domain → by_file_ext → by_date 순으로 소비(6 쿼리)
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
        self.assertEqual(out["by_file_ext"][0], {"file_ext": "pdf", "count": 5})
        self.assertIsNone(out["by_file_ext"][2]["file_ext"])  # 확장자 없음(NULL)
        self.assertEqual(out["by_date"][0], {"date": ts.isoformat(), "count": 10})

    def test_excludes_medical_in_all_queries(self):
        conn = _Conn([(0,), [], [], [], [], []])
        asset_stats(conn)
        # 6개 SQL 모두 의료 제외 WHERE 절을 포함해야 함(total·status·modality·domain·file_ext·date)
        self.assertEqual(len(conn._cur.calls), 6)
        for sql, _params in conn._cur.calls:
            self.assertIn("domain_label <> 'medical'", sql)

    def test_deterministic_order_sql(self):
        conn = _Conn([(0,), [], [], [], [], []])
        asset_stats(conn)
        self.assertIn("ORDER BY COUNT(*) DESC, status ASC", conn._cur.calls[1][0])
        self.assertIn("ORDER BY COUNT(*) DESC, modality ASC", conn._cur.calls[2][0])
        self.assertIn("ORDER BY COUNT(*) DESC, domain_label ASC", conn._cur.calls[3][0])
        self.assertIn("ORDER BY COUNT(*) DESC, ext ASC NULLS LAST", conn._cur.calls[4][0])  # file_ext
        self.assertIn("GROUP BY d ORDER BY d ASC", conn._cur.calls[5][0])  # date 시간순

    def test_period_filter_in_all_queries(self):
        # from/to(생성일 기준·to exclusive) — 6개 집계 모두 기간 반영(프론트 ② 기간별 by_file_ext).
        # dt1<dt2 구간으로 호출(공집합 경계 dt==dt 회피)·파라미터 순서는 [since, until].
        dt1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        dt2 = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([(0,), [], [], [], [], []])
        asset_stats(conn, since=dt1, until=dt2)
        self.assertEqual(len(conn._cur.calls), 6)
        for sql, params in conn._cur.calls:
            self.assertIn("created_at >= %s", sql)
            self.assertIn("created_at < %s", sql)
            self.assertIn("domain_label <> 'medical'", sql)
            self.assertEqual(params, [dt1, dt2])


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

    def test_file_ext_and_date_filters(self):
        dt = datetime(2026, 6, 1, tzinfo=timezone.utc)
        conn = _Conn([(0,), []])
        query_assets(conn, file_ext="pdf", created_from=dt)
        count_sql, count_params = conn._cur.calls[0]
        # file_ext 은 fs_path 확장자 식 = %s, 날짜는 created_at >= %s, 의료 제외 항상 포함
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


class QueryAssetsContentTest(unittest.TestCase):
    """with_content=True — 모달리티 상세 목록에 요약·키워드·제목(파일명) 동반(보완 v6)."""
    def test_with_content_joins_metadata_and_adds_fields(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([
            (1,),
            [("a1", "registered", "video", "general", "/data/raw/뉴스.mp4", ts,
              "서울시장 선거 여론조사 보도", ["선거", "여론조사"])],
        ])
        out = query_assets(conn, modality="video", with_content=True)
        row = out["rows"][0]
        self.assertEqual(row["file_name"], "뉴스.mp4")  # 제목=파일명
        self.assertEqual(row["summary"], "서울시장 선거 여론조사 보도")
        self.assertEqual(row["keywords"], ["선거", "여론조사"])
        # content SELECT 는 asset_metadata LEFT JOIN + ext_meta 요약/키워드
        select_sql = conn._cur.calls[1][0]
        self.assertIn("LEFT JOIN asset_metadata", select_sql)
        self.assertIn("ext_meta", select_sql)
        # JOIN 시 asset_id 는 a. 한정(모호성 방지)
        self.assertIn("ORDER BY a.created_at DESC, a.asset_id DESC", select_sql)

    def test_without_content_keeps_lean_shape(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([(1,), [("a1", "registered", "text", "general", "/d/x.txt", ts)]])
        out = query_assets(conn)  # 기본 with_content=False — 하위호환(요약/키워드 없음)
        self.assertNotIn("summary", out["rows"][0])
        self.assertNotIn("keywords", out["rows"][0])
        self.assertNotIn("LEFT JOIN asset_metadata", conn._cur.calls[1][0])

    def test_with_content_still_excludes_medical(self):
        conn = _Conn([(0,), []])
        query_assets(conn, with_content=True)
        for sql, _p in conn._cur.calls:
            self.assertIn("domain_label <> 'medical'", sql)

    def test_with_content_with_date_qualifies_created_at(self):
        # 🔴 회귀 가드: with_content JOIN + 날짜 필터 시 created_at 모호성(asset·asset_metadata 양쪽 보유)
        # → content SELECT 는 a. 한정해야 PG 오류 없음. COUNT(단일 테이블)은 비한정 유지.
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        conn = _Conn([(0,), []])
        query_assets(conn, with_content=True, created_from=dt, created_to=dt)
        count_sql, select_sql = conn._cur.calls[0][0], conn._cur.calls[1][0]
        self.assertIn("LEFT JOIN asset_metadata", select_sql)
        self.assertIn("a.created_at >= %s", select_sql)  # JOIN 경로 한정
        self.assertIn("a.created_at < %s", select_sql)
        self.assertIn("a.domain_label <> 'medical'", select_sql)
        self.assertNotIn("LEFT JOIN", count_sql)  # COUNT 은 단일 테이블·비한정(모호성 없음)
        self.assertIn("WHERE domain_label <> 'medical'", count_sql)


class ModalityDetailTest(unittest.TestCase):
    """단일 모달리티 스코프 집계(보완 v6) — 확장자·상태·일자 + 총계, 의료 제외."""
    def test_shape_and_modality_bound(self):
        d = datetime(2026, 6, 30, tzinfo=timezone.utc).date()
        # COUNT → by_file_ext → by_status → by_date 순(4 쿼리)
        conn = _Conn([
            (9,),
            [("mp4", 7), ("mov", 2)],
            [("registered", 8), ("failed", 1)],
            [(d, 9)],
        ])
        out = modality_detail(conn, "video")
        self.assertEqual(out["modality"], "video")
        self.assertEqual(out["total"], 9)
        self.assertEqual(out["by_file_ext"][0], {"file_ext": "mp4", "count": 7})
        self.assertEqual(out["by_status"][0], {"status": "registered", "count": 8})
        self.assertEqual(out["by_date"][0], {"date": d.isoformat(), "count": 9})
        # modality 는 %s 바인딩(인젝션 안전)·4 쿼리 모두 의료 제외
        self.assertEqual(len(conn._cur.calls), 4)
        for sql, params in conn._cur.calls:
            self.assertIn("domain_label <> 'medical'", sql)
            self.assertIn("modality = %s", sql)
            self.assertEqual(params, ["video"])

    def test_deterministic_order(self):
        conn = _Conn([(0,), [], [], []])
        modality_detail(conn, "image")
        self.assertIn("ORDER BY COUNT(*) DESC, ext ASC NULLS LAST", conn._cur.calls[1][0])
        self.assertIn("ORDER BY COUNT(*) DESC, status ASC", conn._cur.calls[2][0])
        self.assertIn("GROUP BY d ORDER BY d ASC", conn._cur.calls[3][0])

    def test_period_filter(self):
        # 모달리티 드릴다운도 기간 필터(개요 from/to 와 일관·프론트 ② "모달리티 기간 조회").
        dt1 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        dt2 = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([(0,), [], [], []])
        modality_detail(conn, "video", since=dt1, until=dt2)
        for sql, params in conn._cur.calls:
            self.assertIn("created_at >= %s", sql)
            self.assertIn("created_at < %s", sql)
            self.assertEqual(params, ["video", dt1, dt2])  # modality 먼저, 기간 뒤


class AssetTimelineTest(unittest.TestCase):
    """자산 생성 일자 추이(보완 v6) — group_by 멀티시리즈(계보 timeline 과 동일 패턴)."""
    def test_single_series_default(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([[(ts, 5)]])
        out = asset_timeline(conn, interval="day")
        self.assertEqual(out["interval"], "day")
        self.assertEqual(out["buckets"][0]["count"], 5)
        self.assertNotIn("series", out)
        self.assertIn("domain_label <> 'medical'", conn._cur.calls[0][0])

    def test_group_by_modality_multiseries(self):
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([[("image", ts, 4), ("video", ts, 2)]])
        out = asset_timeline(conn, group_by="modality")
        self.assertEqual(out["group_by"], "modality")
        self.assertEqual([s["key"] for s in out["series"]], ["image", "video"])
        self.assertEqual(out["series"][0]["buckets"][0]["count"], 4)
        sql = conn._cur.calls[0][0]
        self.assertIn("modality AS key", sql)  # 화이트리스트 매핑 컬럼
        self.assertIn("ORDER BY key ASC, bkt ASC", sql)  # 결정적

    def test_group_by_unknown_falls_back_single(self):
        conn = _Conn([[]])
        out = asset_timeline(conn, group_by="evil; DROP TABLE")
        self.assertIn("buckets", out)
        self.assertNotIn("series", out)

    def test_group_by_file_ext_uses_ext_expr(self):
        # 프론트 ③ 일별 파일 포맷 추이 — group_by=file_ext 면 확장자식이 시리즈 key
        ts = datetime(2026, 6, 30, tzinfo=timezone.utc)
        conn = _Conn([[("pdf", ts, 3), ("txt", ts, 1)]])
        out = asset_timeline(conn, group_by="file_ext")
        self.assertEqual(out["group_by"], "file_ext")
        self.assertEqual([s["key"] for s in out["series"]], ["pdf", "txt"])
        sql = conn._cur.calls[0][0]
        self.assertIn("substring(fs_path from", sql)  # 화이트리스트 매핑=확장자 정규식
        self.assertIn("ORDER BY key ASC, bkt ASC", sql)  # 결정적

    def test_bad_interval_falls_back_to_day(self):
        conn = _Conn([[]])
        out = asset_timeline(conn, interval="year")
        self.assertEqual(out["interval"], "day")  # 화이트리스트 폴백(API 는 422 선처리)


if __name__ == "__main__":
    unittest.main()
