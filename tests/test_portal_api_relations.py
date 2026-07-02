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
from types import SimpleNamespace
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


class TestRelationsListFilters(unittest.TestCase):
    """G7 확장(FR-701~753) — relations_list Query 파라미터 파싱·검증·전달."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.list_edges_for_review")
    def test_filters_passed_through(self, mock_list) -> None:
        mock_list.return_value = {"rows": [], "total": 0, "status": "active",
                                  "limit": 50, "offset": 0}
        resp = self.client.get("/admin/relations", params={
            "status": "active", "q": "게임", "asset_id": "as-1", "kind_code": "same_domain",
            "modality": "text", "min_confidence": 0.3, "max_confidence": 0.9,
            "reviewed_by": "bc", "date_on": "reviewed",
            "from": "2026-06-01", "to": "2026-07-01",
        })
        self.assertEqual(resp.status_code, 200)
        kw = mock_list.call_args[1]
        self.assertEqual(kw["q"], "게임")
        self.assertEqual(kw["asset_id"], "as-1")
        self.assertEqual(kw["kind_code"], "same_domain")
        self.assertEqual(kw["modality"], "text")
        self.assertEqual(kw["min_confidence"], 0.3)
        self.assertEqual(kw["max_confidence"], 0.9)
        self.assertEqual(kw["reviewed_by"], "bc")
        self.assertEqual(kw["date_col"], "reviewed_at")  # date_on=reviewed → reviewed_at
        self.assertIsNotNone(kw["since"])
        self.assertIsNotNone(kw["until"])

    @patch("src.app.portal_api.list_edges_for_review")
    def test_no_filters_backward_compatible(self, mock_list) -> None:
        # SC-011 — 확장 파라미터 미지정 시 전부 None(현행 동작). date_col 은 status별 자동.
        mock_list.return_value = {"rows": [], "total": 0, "status": "proposed",
                                  "limit": 50, "offset": 0}
        resp = self.client.get("/admin/relations", params={"status": "proposed"})
        self.assertEqual(resp.status_code, 200)
        kw = mock_list.call_args[1]
        for key in ("q", "asset_id", "kind_code", "modality", "min_confidence",
                    "max_confidence", "reviewed_by", "since", "until"):
            self.assertIsNone(kw[key], key)
        self.assertEqual(kw["date_col"], "created_at")  # proposed → created_at

    @patch("src.app.portal_api.list_edges_for_review")
    def test_date_col_auto_by_status(self, mock_list) -> None:
        # FR-752 — date_on 생략 시 active/rejected → reviewed_at.
        mock_list.return_value = {"rows": [], "total": 0, "status": "active",
                                  "limit": 50, "offset": 0}
        resp = self.client.get("/admin/relations", params={"status": "active"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_list.call_args[1]["date_col"], "reviewed_at")

    @patch("src.app.portal_api.list_edges_for_review")
    def test_blank_q_ignored(self, mock_list) -> None:
        # 빈/공백 q → None(필터 비활성·팀 결정).
        mock_list.return_value = {"rows": [], "total": 0, "status": "proposed",
                                  "limit": 50, "offset": 0}
        resp = self.client.get("/admin/relations", params={"status": "proposed", "q": "   "})
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(mock_list.call_args[1]["q"])

    @patch("src.app.portal_api.list_edges_for_review")
    def test_min_greater_than_max_400(self, mock_list) -> None:
        resp = self.client.get("/admin/relations", params={
            "status": "proposed", "min_confidence": 0.9, "max_confidence": 0.1})
        self.assertEqual(resp.status_code, 400)
        mock_list.assert_not_called()

    @patch("src.app.portal_api.list_edges_for_review")
    def test_confidence_out_of_range_400(self, mock_list) -> None:
        resp = self.client.get("/admin/relations", params={
            "status": "proposed", "min_confidence": 1.5})
        self.assertEqual(resp.status_code, 400)
        resp2 = self.client.get("/admin/relations", params={
            "status": "proposed", "max_confidence": -0.2})
        self.assertEqual(resp2.status_code, 400)
        mock_list.assert_not_called()

    @patch("src.app.portal_api.list_edges_for_review")
    def test_bogus_date_on_400(self, mock_list) -> None:
        resp = self.client.get("/admin/relations", params={
            "status": "proposed", "date_on": "bogus"})
        self.assertEqual(resp.status_code, 400)
        mock_list.assert_not_called()

    @patch("src.app.portal_api.list_edges_for_review")
    def test_bad_date_format_422(self, mock_list) -> None:
        # 013 _parse_dt 관례 — 형식 오류는 422.
        resp = self.client.get("/admin/relations", params={
            "status": "proposed", "from": "not-a-date"})
        self.assertEqual(resp.status_code, 422)
        mock_list.assert_not_called()

    @patch("src.app.portal_api.list_edges_for_review")
    def test_from_after_to_400(self, mock_list) -> None:
        resp = self.client.get("/admin/relations", params={
            "status": "proposed", "from": "2026-07-01", "to": "2026-06-01"})
        self.assertEqual(resp.status_code, 400)
        mock_list.assert_not_called()

    @patch("src.app.portal_api.list_edges_for_review")
    def test_q_over_max_length_422(self, mock_list) -> None:
        # FR-702 — q 는 최대 200자. Query(max_length=200) 초과 시 FastAPI 검증 422.
        resp = self.client.get("/admin/relations", params={
            "status": "proposed", "q": "x" * 201})
        self.assertEqual(resp.status_code, 422)
        mock_list.assert_not_called()


class TestRelationKindsList(unittest.TestCase):
    """G7 확장(FR-801) — GET /admin/relation-kinds 목록·status 화이트리스트·401."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.list_relation_kinds")
    def test_list_all(self, mock_kinds) -> None:
        mock_kinds.return_value = {"rows": [
            {"kind_code": "same_domain", "kind_name_ko": "동일 도메인", "status": "active"}],
            "total": 1}
        resp = self.client.get("/admin/relation-kinds")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_kinds.call_args[1]["status"], None)
        self.assertEqual(resp.json()["total"], 1)

    @patch("src.app.portal_api.list_relation_kinds")
    def test_list_status_active(self, mock_kinds) -> None:
        mock_kinds.return_value = {"rows": [], "total": 0}
        resp = self.client.get("/admin/relation-kinds", params={"status": "active"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(mock_kinds.call_args[1]["status"], "active")

    @patch("src.app.portal_api.list_relation_kinds")
    def test_list_bogus_status_400(self, mock_kinds) -> None:
        resp = self.client.get("/admin/relation-kinds", params={"status": "bogus"})
        self.assertEqual(resp.status_code, 400)
        mock_kinds.assert_not_called()


class TestRelationKindsListAuth(unittest.TestCase):
    """인증 없음 → 401."""

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

    def test_kinds_without_token_401(self) -> None:
        resp = self.client.get("/admin/relation-kinds")
        self.assertEqual(resp.status_code, 401)


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
        # 055 FR-201: approve/reject 와 동일 봉투 {results:[{edge_id,ok}]} 로 통일
        self.assertEqual(resp.json(), {"results": [{"edge_id": "e1", "ok": True}]})
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
        self.assertEqual(resp.json(), {"results": [{"edge_id": "missing", "ok": False}]})
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


class TestReviewTopicReindexHook(unittest.TestCase):
    """056 FR-301~304 — 검토 결정(승인/반려/정정) 커밋 후 OS topics 재색인 훅.

    검증 의도
        - ``ok=True`` 엣지만 양끝 자산으로 해소해 ``reindex_asset_topics`` 를 호출한다(트랜잭션 밖).
        - **best-effort(FR-304)**: 재색인이 raise 해도 핸들러는 정상 응답을 그대로 돌려준다.
        - ``ok=True`` 가 없거나 OS 동기화 off 면 재색인을 호출하지 않는다.
    ``reindex_asset_topics``·엣지 양끝 해소·설정·PostgresUtil 을 patch 해 실 DB·OS 없이 단위 검증.
    """

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    # 승인 — ok=True 엣지(e1)만 양끝 해소 → reindex 호출, 응답은 재색인과 무관하게 동일.
    @patch("src.database.postgres_util.PostgresUtil", return_value=MagicMock())
    @patch("src.app.portal_api.get_current_settings",
           return_value=SimpleNamespace(opensearch_sync_enabled=True))
    @patch("src.app.portal_api.reindex_asset_topics")
    @patch("src.app.portal_api._resolve_edge_endpoint_assets", return_value=["A", "B"])
    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.bulk_review")
    def test_approve_reindexes_ok_edge_endpoints(
        self, m_bulk, m_audit, m_resolve, m_reindex, m_settings, m_pgutil
    ) -> None:
        m_bulk.return_value = [{"edge_id": "e1", "ok": True}, {"edge_id": "e2", "ok": False}]
        resp = self.client.post("/admin/relations/approve", json={"edge_ids": ["e1", "e2"]})
        self.assertEqual(resp.status_code, 200)
        # 응답 봉투는 종전과 동일(재색인은 응답에 영향 없음).
        self.assertEqual(resp.json(),
                         {"results": [{"edge_id": "e1", "ok": True},
                                      {"edge_id": "e2", "ok": False}]})
        # ok=True(e1)만 양끝 해소 대상. ok=False(e2) 제외.
        self.assertEqual(m_resolve.call_args.args[0], ["e1"])
        # 해소된 양끝 자산(A,B)으로 재색인 1회.
        m_reindex.assert_called_once()
        self.assertEqual(m_reindex.call_args.kwargs["asset_ids"], ["A", "B"])

    # best-effort(FR-304): 재색인이 raise 해도 승인 응답은 정상.
    @patch("src.database.postgres_util.PostgresUtil", return_value=MagicMock())
    @patch("src.app.portal_api.get_current_settings",
           return_value=SimpleNamespace(opensearch_sync_enabled=True))
    @patch("src.app.portal_api.reindex_asset_topics", side_effect=RuntimeError("OS down"))
    @patch("src.app.portal_api._resolve_edge_endpoint_assets", return_value=["A"])
    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.bulk_review")
    def test_approve_reindex_failure_does_not_break_response(
        self, m_bulk, m_audit, m_resolve, m_reindex, m_settings, m_pgutil
    ) -> None:
        m_bulk.return_value = [{"edge_id": "e1", "ok": True}]
        resp = self.client.post("/admin/relations/approve", json={"edge_ids": ["e1"]})
        self.assertEqual(resp.status_code, 200)  # 재색인 예외에도 승인 성공
        self.assertEqual(resp.json(), {"results": [{"edge_id": "e1", "ok": True}]})

    # ok=True 가 없으면 재색인·해소 모두 미호출(변경 없음).
    @patch("src.app.portal_api.reindex_asset_topics")
    @patch("src.app.portal_api._resolve_edge_endpoint_assets")
    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.bulk_review")
    def test_approve_all_ok_false_no_reindex(
        self, m_bulk, m_audit, m_resolve, m_reindex
    ) -> None:
        m_bulk.return_value = [{"edge_id": "e1", "ok": False}]
        resp = self.client.post("/admin/relations/approve", json={"edge_ids": ["e1"]})
        self.assertEqual(resp.status_code, 200)
        m_resolve.assert_not_called()
        m_reindex.assert_not_called()

    # OS 동기화 off → 재색인 스킵(020/038 게이트). 응답은 정상.
    @patch("src.app.portal_api.get_current_settings",
           return_value=SimpleNamespace(opensearch_sync_enabled=False))
    @patch("src.app.portal_api.reindex_asset_topics")
    @patch("src.app.portal_api._resolve_edge_endpoint_assets")
    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.bulk_review")
    def test_approve_sync_disabled_skips_reindex(
        self, m_bulk, m_audit, m_resolve, m_reindex, m_settings
    ) -> None:
        m_bulk.return_value = [{"edge_id": "e1", "ok": True}]
        resp = self.client.post("/admin/relations/approve", json={"edge_ids": ["e1"]})
        self.assertEqual(resp.status_code, 200)
        m_resolve.assert_not_called()
        m_reindex.assert_not_called()

    # 정정(revise) 성공(ok=True) → 그 엣지 양끝 재색인. 응답 봉투 동일.
    @patch("src.database.postgres_util.PostgresUtil", return_value=MagicMock())
    @patch("src.app.portal_api.get_current_settings",
           return_value=SimpleNamespace(opensearch_sync_enabled=True))
    @patch("src.app.portal_api.reindex_asset_topics")
    @patch("src.app.portal_api._resolve_edge_endpoint_assets", return_value=["A", "B"])
    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.revise_edge")
    def test_revise_reindexes_when_changed(
        self, m_revise, m_audit, m_resolve, m_reindex, m_settings, m_pgutil
    ) -> None:
        m_revise.return_value = True
        resp = self.client.post("/admin/relations/revise",
                                json={"edge_id": "e1", "to_status": "rejected"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"results": [{"edge_id": "e1", "ok": True}]})
        self.assertEqual(m_resolve.call_args.args[0], ["e1"])
        m_reindex.assert_called_once()
        self.assertEqual(m_reindex.call_args.kwargs["asset_ids"], ["A", "B"])

    # 정정 실패(ok=False·변경 없음) → 재색인 미호출.
    @patch("src.app.portal_api.reindex_asset_topics")
    @patch("src.app.portal_api._resolve_edge_endpoint_assets")
    @patch("src.app.portal_api.record_access")
    @patch("src.app.portal_api.revise_edge")
    def test_revise_ok_false_no_reindex(
        self, m_revise, m_audit, m_resolve, m_reindex
    ) -> None:
        m_revise.return_value = False
        resp = self.client.post("/admin/relations/revise",
                                json={"edge_id": "missing", "to_status": "active"})
        self.assertEqual(resp.status_code, 200)
        m_resolve.assert_not_called()
        m_reindex.assert_not_called()
