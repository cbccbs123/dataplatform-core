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
            r = self.client.get("/assets/a1/lineage")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["activities"][0]["activity"], "ingest.received.v1")

    def test_access_logs_endpoint(self):
        with mock.patch.object(portal_api, "query_access_logs",
                               return_value={"rows": [], "total": 0}), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/access-logs?action=search")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"rows": [], "total": 0})

    def test_stats_endpoint(self):
        with mock.patch.object(portal_api, "access_log_stats",
                               return_value={"total": 0, "by_action": [], "by_user": []}), \
             mock.patch.object(portal_api, "_run_in_db", side_effect=lambda cb: cb(None)):
            r = self.client.get("/access-logs/stats")
        self.assertEqual(r.status_code, 200)
        self.assertIn("by_action", r.json())

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
            portal_api._record_access_safe("GET", "/access-logs", 200, "u1")  # 감사 뷰(자기 기록 안 함)
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
