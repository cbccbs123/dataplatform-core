"""052 HITL 관계 검토 포탈 API 단위 테스트 — DB·LLM·네트워크 불필요.

전략(052 plan G3/G4)
    FastAPI ``TestClient`` 로 라우팅·status/to_status 화이트리스트·per-id 결과·401·감사 호출만
    검증한다. ``review.py`` 소비 함수(``list_edges_for_review``/``bulk_review``/``revise_edge``/
    ``promote_relation_kind``)와 DB 실행 seam(``_run_in_db``/``_run_in_db_write``)·감사
    (``record_access``)를 ``unittest.mock.patch`` 로 대체해 순수 단위로 돈다.

    ``tests/test_portal_api.py`` 의 auth bypass + passthrough DB 패턴을 그대로 따른다.
"""
from __future__ import annotations

import os
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.app.portal_api import app

_AUTH_DISABLED_ENV = {
    "PORTAL_AUTH_DISABLED": "1",
    "PORTAL_JWT_SECRET": "test-secret",
}


class _FakeConn:
    """passthrough conn 대역 — ``conn.transaction()`` 을 no-op 컨텍스트로 지원한다.

    ``_record_relation_audit`` 이 감사를 savepoint(``with conn.transaction():``)로 감싸므로,
    write 핸들러 단위 테스트의 conn 은 이 컨텍스트를 지원해야 한다(record_access 는 patch).
    """

    @contextmanager
    def transaction(self):
        yield self


def _passthrough_db(callback):
    """``_run_in_db``/``_run_in_db_write`` 대역 — fake conn 으로 callback 을 즉시 실행."""
    return callback(_FakeConn())


def _enable_bypass(test_case: unittest.TestCase) -> None:
    """dev auth bypass + 조회/쓰기 DB seam passthrough."""
    env = patch.dict(os.environ, _AUTH_DISABLED_ENV, clear=False)
    env.start()
    test_case.addCleanup(env.stop)
    for target in ("_run_in_db", "_run_in_db_write"):
        p = patch(f"src.app.portal_api.{target}", _passthrough_db)
        p.start()
        test_case.addCleanup(p.stop)


class TestRelationsList(unittest.TestCase):
    """GET /admin/relations — 조회 위임·status 화이트리스트·401."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.list_edges_for_review")
    def test_list_passes_status_limit_offset(self, mock_list) -> None:
        mock_list.return_value = {"rows": [], "total": 0, "status": "proposed",
                                  "limit": 50, "offset": 0}
        resp = self.client.get("/admin/relations",
                               params={"status": "proposed", "limit": 50, "offset": 10})
        self.assertEqual(resp.status_code, 200)
        _conn, kwargs = mock_list.call_args[0], mock_list.call_args[1]
        self.assertEqual(kwargs["status"], "proposed")
        self.assertEqual(kwargs["limit"], 50)
        self.assertEqual(kwargs["offset"], 10)
        self.assertEqual(resp.json()["status"], "proposed")

    @patch("src.app.portal_api.list_edges_for_review")
    def test_list_default_status_proposed(self, mock_list) -> None:
        mock_list.return_value = {"rows": [], "total": 0, "status": "proposed",
                                  "limit": 50, "offset": 0}
        resp = self.client.get("/admin/relations")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_list.call_args[1]["status"], "proposed")

    @patch("src.app.portal_api.list_edges_for_review")
    def test_list_bogus_status_400(self, mock_list) -> None:
        resp = self.client.get("/admin/relations", params={"status": "bogus"})
        self.assertEqual(resp.status_code, 400)
        mock_list.assert_not_called()


class TestRelationsListAuth(unittest.TestCase):
    """인증 없음(bypass off·토큰 없음) → 401."""

    def setUp(self) -> None:
        from src.portal.auth.verifier import _reset_verifier_for_tests
        _reset_verifier_for_tests()
        self._env = patch.dict(
            os.environ, {"PORTAL_AUTH_DISABLED": "0", "PORTAL_JWT_SECRET": "test-secret"},
            clear=False)
        self._env.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        from src.portal.auth.verifier import _reset_verifier_for_tests
        self._env.stop()
        _reset_verifier_for_tests()

    def test_list_without_token_401(self) -> None:
        resp = self.client.get("/admin/relations", params={"status": "proposed"})
        self.assertEqual(resp.status_code, 401)

    def test_approve_without_token_401(self) -> None:
        resp = self.client.post("/admin/relations/approve", json={"edge_ids": ["e1"]})
        self.assertEqual(resp.status_code, 401)


class TestRelationsApproveReject(unittest.TestCase):
    """POST /admin/relations/{approve,reject} — per-id 결과·reviewer·감사(FR-201~203/502)."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.bulk_review")
    def test_approve_returns_results_and_audits_ok(self, mock_bulk, mock_audit) -> None:
        mock_bulk.return_value = [{"edge_id": "e1", "ok": True}, {"edge_id": "e2", "ok": False}]
        resp = self.client.post("/admin/relations/approve",
                                json={"edge_ids": ["e1", "e2"]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(),
                         {"results": [{"edge_id": "e1", "ok": True},
                                      {"edge_id": "e2", "ok": False}]})
        # bulk_review 는 action=approve·reviewer=principal.user_id(bypass=anonymous)
        self.assertEqual(mock_bulk.call_args[1]["action"], "approve")
        self.assertEqual(mock_bulk.call_args[1]["edge_ids"], ["e1", "e2"])
        self.assertEqual(mock_bulk.call_args[1]["reviewer"], "anonymous")
        # 감사는 ok=True 건(e1)만 relation.approve 로 기록·ok=False(e2)는 미기록
        approve_calls = [c for c in mock_audit.call_args_list
                         if c.kwargs.get("action") == "relation.approve"]
        self.assertEqual(len(approve_calls), 1)
        self.assertEqual(approve_calls[0].kwargs["user_id"], "anonymous")
        self.assertEqual(approve_calls[0].kwargs["detail"], {"edge_id": "e1"})

    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.bulk_review")
    def test_reject_dispatches_reject_action(self, mock_bulk, mock_audit) -> None:
        mock_bulk.return_value = [{"edge_id": "e1", "ok": True}]
        resp = self.client.post("/admin/relations/reject", json={"edge_ids": ["e1"]})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_bulk.call_args[1]["action"], "reject")
        actions = [c.kwargs.get("action") for c in mock_audit.call_args_list]
        self.assertIn("relation.reject", actions)

    @patch("src.app.portal_api.bulk_review")
    def test_empty_edge_ids_400(self, mock_bulk) -> None:
        resp = self.client.post("/admin/relations/approve", json={"edge_ids": []})
        self.assertEqual(resp.status_code, 400)
        mock_bulk.assert_not_called()


class TestRelationsRevise(unittest.TestCase):
    """POST /admin/relations/revise — 사람 전용 정정·to_status 화이트리스트·감사(FR-301/502)."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.revise_edge")
    def test_revise_calls_and_audits(self, mock_revise, mock_audit) -> None:
        mock_revise.return_value = True
        resp = self.client.post("/admin/relations/revise",
                                json={"edge_id": "e1", "to_status": "rejected"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"edge_id": "e1", "ok": True})
        self.assertEqual(mock_revise.call_args[1]["edge_id"], "e1")
        self.assertEqual(mock_revise.call_args[1]["to_status"], "rejected")
        self.assertEqual(mock_revise.call_args[1]["reviewer"], "anonymous")
        revise_calls = [c for c in mock_audit.call_args_list
                        if c.kwargs.get("action") == "relation.revise"]
        self.assertEqual(len(revise_calls), 1)
        self.assertEqual(revise_calls[0].kwargs["detail"],
                         {"edge_id": "e1", "to_status": "rejected"})

    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.revise_edge")
    def test_revise_ok_false_no_audit(self, mock_revise, mock_audit) -> None:
        mock_revise.return_value = False
        resp = self.client.post("/admin/relations/revise",
                                json={"edge_id": "missing", "to_status": "active"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"edge_id": "missing", "ok": False})
        revise_calls = [c for c in mock_audit.call_args_list
                        if c.kwargs.get("action") == "relation.revise"]
        self.assertEqual(len(revise_calls), 0)

    @patch("src.app.portal_api.revise_edge")
    def test_revise_bogus_to_status_400(self, mock_revise) -> None:
        resp = self.client.post("/admin/relations/revise",
                                json={"edge_id": "e1", "to_status": "bogus"})
        self.assertEqual(resp.status_code, 400)
        mock_revise.assert_not_called()


class TestRelationKindPromote(unittest.TestCase):
    """POST /admin/relation-kinds/{code}/promote — 종류 승격·감사(FR-401/502)."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.promote_relation_kind")
    def test_promote_calls_and_audits(self, mock_promote, mock_audit) -> None:
        mock_promote.return_value = True
        resp = self.client.post("/admin/relation-kinds/gaming_hardware/promote")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"kind_code": "gaming_hardware", "ok": True})
        self.assertEqual(mock_promote.call_args[1]["kind_code"], "gaming_hardware")
        self.assertEqual(mock_promote.call_args[1]["reviewer"], "anonymous")
        promote_calls = [c for c in mock_audit.call_args_list
                         if c.kwargs.get("action") == "relation.kind_promote"]
        self.assertEqual(len(promote_calls), 1)
        self.assertEqual(promote_calls[0].kwargs["detail"], {"kind_code": "gaming_hardware"})

    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.promote_relation_kind")
    def test_promote_ok_false_no_audit(self, mock_promote, mock_audit) -> None:
        mock_promote.return_value = False
        resp = self.client.post("/admin/relation-kinds/already_active/promote")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"kind_code": "already_active", "ok": False})
        promote_calls = [c for c in mock_audit.call_args_list
                         if c.kwargs.get("action") == "relation.kind_promote"]
        self.assertEqual(len(promote_calls), 0)


class TestRelationAuditBestEffort(unittest.TestCase):
    """FR-502 — 감사 기록 실패가 결정 트랜잭션을 깨지 않는다(best-effort·savepoint)."""

    def test_record_relation_audit_swallows_failure(self) -> None:
        from src.app.portal_api import _record_relation_audit
        conn = MagicMock()
        # conn.transaction() 컨텍스트 진입 시 예외 → best-effort 로 삼켜야 한다.
        conn.transaction.side_effect = RuntimeError("db down")
        # 예외를 전파하지 않으면 성공(결정 트랜잭션 보존).
        _record_relation_audit(conn, action="relation.approve", reviewer="bc",
                               detail={"edge_id": "e1"})
