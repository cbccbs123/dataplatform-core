import os
import unittest
from unittest import mock

os.environ.setdefault("PORTAL_AUTH_DISABLED", "1")  # dev bypass(anonymous=public)

from fastapi.testclient import TestClient  # noqa: E402

from src.app import portal_api  # noqa: E402


class HistoryEndpointsTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(portal_api.app)

    def test_lineage_endpoint(self):
        with mock.patch.object(portal_api, "query_asset_lineage",
                               return_value=[{"activity": "ingest.received.v1", "agent": "run_ingest",
                                              "used": {}, "generated": {}, "occurred_at": "2026-06-30T00:00:00+00:00"}]), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/admin/assets/a1/lineage")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["activities"][0]["activity"], "ingest.received.v1")

    def test_access_logs_endpoint(self):
        with mock.patch.object(portal_api, "query_access_logs",
                               return_value={"rows": [], "total": 0}), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/admin/access-logs?action=search")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"rows": [], "total": 0})

    def test_stats_endpoint(self):
        with mock.patch.object(portal_api, "access_log_stats",
                               return_value={"total": 0, "by_action": [], "by_user": []}), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/admin/access-logs/stats")
        self.assertEqual(r.status_code, 200)
        self.assertIn("by_action", r.json())

    def test_access_logs_bad_date_returns_422(self):
        # _parse_dt: 잘못된 날짜 형식 → HTTPException(422)(헌법 8조·오류 경로 검증).
        r = self.client.get("/admin/access-logs?from=not-a-date")
        self.assertEqual(r.status_code, 422)

    def test_lineage_feed_endpoint(self):
        with mock.patch.object(portal_api, "query_lineage_feed",
                               return_value={"rows": [{"lineage_id": "l1", "asset_id": "a1",
                                                       "activity": "ingest.registered.v1", "agent": "run_ingest",
                                                       "occurred_at": "2026-06-30T00:00:00+00:00"}], "total": 1}) as feed, \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/admin/lineage?limit=10&modality=video&status=registered&file_ext=mp4")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["total"], 1)
        self.assertEqual(r.json()["rows"][0]["asset_id"], "a1")
        # 라우터→서비스 자산차원 필터 배선 검증(modality·status·file_ext 전달).
        kw = feed.call_args.kwargs
        self.assertEqual((kw["modality"], kw["status"], kw["file_ext"]), ("video", "registered", "mp4"))

    def test_timeline_endpoint(self):
        with mock.patch.object(portal_api, "access_log_timeline",
                               return_value={"interval": "day", "buckets": [{"bucket": "2026-06-30T00:00:00+00:00", "count": 5}]}), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/admin/access-logs/timeline?interval=day&action=search")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["buckets"][0]["count"], 5)

    def test_timeline_bad_interval_422(self):
        r = self.client.get("/admin/access-logs/timeline?interval=year")
        self.assertEqual(r.status_code, 422)

    def test_asset_stats_endpoint(self):
        with mock.patch.object(portal_api, "asset_stats",
                               return_value={"total": 3, "by_status": [{"status": "registered", "count": 3}],
                                             "by_modality": [], "by_domain": [],
                                             "by_file_ext": [{"file_ext": "pdf", "count": 3}],
                                             "by_date": [{"date": "2026-06-30", "count": 3}]}), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/admin/asset-stats")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["by_status"][0]["status"], "registered")
        # 신규 차원(file_ext·date)도 응답에 그대로 실린다(API 레벨).
        self.assertIn("by_file_ext", body)
        self.assertIn("by_date", body)

    def test_assets_list_endpoint(self):
        with mock.patch.object(portal_api, "query_assets",
                               return_value={"rows": [{"asset_id": "a1", "status": "registered",
                                                       "modality": "text", "domain_label": "general",
                                                       "file_name": "x.txt", "created_at": "2026-06-30T00:00:00+00:00"}],
                                             "total": 1}), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/admin/assets?status=registered&limit=10")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["rows"][0]["status"], "registered")

    def test_record_access_safe_records_data_route(self):
        # 기록 결정 로직 직접 검증(미들웨어 fire-and-forget 타이밍과 무관·결정적):
        # 데이터 라우트 성공 응답 → record_access(action=asset_view·asset_id) 1회.
        with mock.patch.object(portal_api, "_run_in_db_write", side_effect=lambda cb: cb(None)), \
             mock.patch.object(portal_api, "record_access") as rec:
            portal_api._record_access_safe("GET", "/assets/a1", 200, "u1")
        rec.assert_called_once()
        self.assertEqual(rec.call_args.kwargs["action"], "asset_view")
        self.assertEqual(rec.call_args.kwargs["asset_id"], "a1")
        self.assertEqual(rec.call_args.kwargs["user_id"], "u1")

    def test_record_access_safe_skips_non_data_and_error_status(self):
        with mock.patch.object(portal_api, "record_access") as rec:
            portal_api._record_access_safe("GET", "/health", 200, "u1")     # 비대상 라우트
            portal_api._record_access_safe("GET", "/assets/a1", 404, "u1")  # 4xx
            portal_api._record_access_safe("GET", "/admin/access-logs", 200, "u1")  # 감사 뷰(자기 기록 안 함)
            # 신규 관리자/대시보드 뷰(/admin/*)도 자기 기록 안 함(노이즈 방지·통합 경로 검증).
            portal_api._record_access_safe("GET", "/admin/lineage", 200, "u1")
            portal_api._record_access_safe("GET", "/admin/access-logs/timeline", 200, "u1")
            portal_api._record_access_safe("GET", "/admin/asset-stats", 200, "u1")
            portal_api._record_access_safe("GET", "/admin/assets", 200, "u1")
        rec.assert_not_called()

    def test_middleware_schedules_recording_non_blocking(self):
        # 미들웨어는 기록을 await 하지 않고 create_task 로 스케줄(비차단)·응답은 그대로 반환.
        # _record_access_bg 를 AsyncMock 으로 가로채 호출 인자만 확인(실 DB·실제 태스크 실행 불요).
        with mock.patch.object(portal_api, "fetch_asset_detail", return_value={"asset_id": "a1"}), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)), \
             mock.patch.object(portal_api, "_record_access_bg", new=mock.AsyncMock()) as bg:
            r = self.client.get("/assets/a1")
        self.assertEqual(r.status_code, 200)          # 응답 정상(기록과 분리)
        bg.assert_called_once()                       # 기록은 스케줄됨
        self.assertEqual(bg.call_args.args[0], "GET")
        self.assertEqual(bg.call_args.args[1], "/assets/a1")


if __name__ == "__main__":
    unittest.main()
