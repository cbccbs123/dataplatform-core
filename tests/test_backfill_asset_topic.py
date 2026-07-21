"""065 T401 — 자산 자기주제 백필 CLI 단위 테스트(mock·DB/LLM 불필요).

검증 의도 (FR-501·FR-503 · plan G4)
    ``run_topic_backfill`` 은 registered + 메타 보유 자산의 저장된 summary/keywords 로
    ``classify_asset_topic`` 을 일괄 호출해 ``asset_topic`` 정본을 소급 부여한다(재수집 불요).
    실 DB/LLM 없이 다음을 덮는다:
      - 대상 스캔 SQL 형상(registered + 메타 EXISTS · ``--only-missing`` LEFT JOIN IS NULL · 결정적 정렬)
      - 자산별 격리(한 건 예외에도 배치 계속·failed 카운트)
      - 재실행 멱등(이미 부여된 자산 스킵·재분류 0)
      - 요약 카운트({scanned, classified, skipped_existing, no_text, failed, os_synced})
      - OS 재색인은 topic 이 생긴 자산만·``--no-os-sync``(os_sync_fn=None) 로 끔
    classify/os_sync/has_topic 는 seam 주입으로 대체해 순수 단위로 분기만 검증한다.
"""
from __future__ import annotations

import unittest
import uuid
from unittest.mock import MagicMock

from scripts.backfill_asset_topic import (
    _asset_has_topic,
    _fetch_target_asset_ids,
    backfill_assets,
    build_status_report,
    format_status_lines,
)


def _mock_conn(*, fetchall_val=None, fetchone_val=None):
    """``conn.cursor(...)`` 컨텍스트매니저 mock — fetchall/fetchone 주입값 반환."""
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchall.return_value = fetchall_val if fetchall_val is not None else []
    cur.fetchone.return_value = fetchone_val
    conn.cursor.return_value = cur
    return conn, cur


class _FakeDB:
    """``execute_in_transaction`` 이 콜백을 주입 conn 으로 즉시 실행하는 PostgresUtil 최소 대역."""

    def __init__(self, conn=None):
        self._conn = conn or MagicMock()

    def execute_in_transaction(self, callback, *, idempotent=False):
        return callback(self._conn)


# ────────────────────────────────────────────────────────────────────────────
# 1) 대상 스캔 SQL 형상 (_fetch_target_asset_ids)
# ────────────────────────────────────────────────────────────────────────────
class TestFetchTargets(unittest.TestCase):
    def test_only_missing_left_join_is_null(self):
        conn, cur = _mock_conn(fetchall_val=[("a1",), ("a2",)])
        ids = _fetch_target_asset_ids(conn, only_missing=True)
        self.assertEqual(ids, ["a1", "a2"])
        sql = cur.execute.call_args.args[0].lower()
        # 미부여만: asset_topic LEFT JOIN + IS NULL, registered, 메타 EXISTS, 결정적 정렬.
        self.assertIn("left join asset_topic", sql)
        self.assertIn("is null", sql)
        self.assertIn("status = 'registered'", sql)
        self.assertIn("exists", sql)
        self.assertIn("order by a.asset_id", sql)

    def test_all_mode_has_no_missing_filter(self):
        conn, cur = _mock_conn(fetchall_val=[])
        _fetch_target_asset_ids(conn, only_missing=False)
        sql = cur.execute.call_args.args[0].lower()
        self.assertNotIn("left join asset_topic", sql)
        self.assertIn("exists", sql)  # 메타 보유 조건은 유지
        self.assertIn("status = 'registered'", sql)

    def test_limit_applied_as_param(self):
        conn, cur = _mock_conn(fetchall_val=[])
        _fetch_target_asset_ids(conn, only_missing=True, limit=10)
        sql = cur.execute.call_args.args[0].lower()
        self.assertIn("limit", sql)
        self.assertEqual(cur.execute.call_args.args[1], (10,))

    def test_asset_id_str_coercion(self):
        u = uuid.uuid4()
        conn, _ = _mock_conn(fetchall_val=[(u,)])
        self.assertEqual(_fetch_target_asset_ids(conn, only_missing=True), [str(u)])


class TestAssetHasTopic(unittest.TestCase):
    def test_true_when_row_present(self):
        conn, _ = _mock_conn(fetchone_val=(1,))
        self.assertTrue(_asset_has_topic(conn, "a1"))

    def test_false_when_no_row(self):
        conn, _ = _mock_conn(fetchone_val=None)
        self.assertFalse(_asset_has_topic(conn, "a1"))


# ────────────────────────────────────────────────────────────────────────────
# 2) 배치 루프 (backfill_assets) — seam 주입으로 분기만 검증
# ────────────────────────────────────────────────────────────────────────────
class TestBackfillLoop(unittest.TestCase):
    def _run(self, *, ids, classify_fn, os_sync_fn=None, skip_existing=False,
             has_topic_fn=None):
        db = _FakeDB()
        return backfill_assets(
            db, ids, classify_fn=classify_fn, os_sync_fn=os_sync_fn,
            skip_existing=skip_existing,
            has_topic_fn=has_topic_fn or (lambda conn, aid: False),
            log_every=0,
        )

    def test_counts_classified_vs_none(self):
        # a1 → dict(부여), a2 → None(미부여·no_text 버킷).
        def classify(conn, aid, *, settings=None, client=None):
            return {"topic_ko": "스포츠·레저"} if aid == "a1" else None

        s = self._run(ids=["a1", "a2"], classify_fn=classify)
        self.assertEqual(s["scanned"], 2)
        self.assertEqual(s["classified"], 1)
        self.assertEqual(s["no_text"], 1)
        self.assertEqual(s["failed"], 0)
        self.assertEqual(s["os_synced"], 0)

    def test_isolation_one_exception_continues(self):
        def classify(conn, aid, *, settings=None, client=None):
            if aid == "a2":
                raise RuntimeError("boom")
            return {"topic_ko": "과학"}

        s = self._run(ids=["a1", "a2", "a3"], classify_fn=classify)
        self.assertEqual(s["scanned"], 3)
        self.assertEqual(s["classified"], 2)  # a1, a3
        self.assertEqual(s["failed"], 1)  # a2 격리

    def test_rerun_skips_existing_without_classify(self):
        # skip_existing + has_topic True → 재분류 안 함(멱등). classify 미호출.
        classify = MagicMock(return_value={"topic_ko": "과학"})
        s = self._run(
            ids=["a1", "a2"], classify_fn=classify,
            skip_existing=True, has_topic_fn=lambda conn, aid: True,
        )
        self.assertEqual(s["skipped_existing"], 2)
        self.assertEqual(s["classified"], 0)
        classify.assert_not_called()

    def test_os_sync_counts_only_on_classified(self):
        def classify(conn, aid, *, settings=None, client=None):
            return None if aid == "a3" else {"topic_ko": "과학"}

        synced: list = []

        def os_sync(aid):
            synced.append(aid)
            return True

        s = self._run(ids=["a1", "a2", "a3"], classify_fn=classify, os_sync_fn=os_sync)
        self.assertEqual(s["classified"], 2)
        self.assertEqual(s["os_synced"], 2)  # a1, a2 (a3 미부여 → 색인 안 함)
        self.assertEqual(synced, ["a1", "a2"])

    def test_no_os_sync_when_fn_none(self):
        def classify(conn, aid, *, settings=None, client=None):
            return {"topic_ko": "과학"}

        s = self._run(ids=["a1"], classify_fn=classify, os_sync_fn=None)
        self.assertEqual(s["os_synced"], 0)


# ────────────────────────────────────────────────────────────────────────────
# 2b) 품질 재백필 모드(--reclassify · T604 · FR-704)
# ────────────────────────────────────────────────────────────────────────────
class TestReclassifyMode(unittest.TestCase):
    """T604 — ``--reclassify``: 기존 asset_topic 행도 재분류 대상. None→행 삭제(미부여 전이)·dict→upsert.

    재수집 검증서 드러난 3결함(placeholder 미분류·미분류 catch-all·과편화) 반영 후, 고정 레지스트리로
    전 자산을 동일 기준으로 재분류해 기존 저장분을 정리한다. 스캔은 미부여 필터를 풀어(only_missing=
    False) 이미 부여된 행도 포함하고, 재분류 결과가 None(미부여)이면 기존 행을 삭제한다.
    """

    def _run(self, *, ids, classify_fn, delete_fn=None, reclassify=True):
        db = _FakeDB()
        return backfill_assets(
            db, ids, classify_fn=classify_fn,
            reclassify=reclassify, delete_fn=delete_fn,
            skip_existing=False, has_topic_fn=lambda conn, aid: False,
            log_every=0,
        )

    def test_reclassify_scan_includes_existing_rows(self):
        # 재분류는 미부여 필터를 풀어 기존 행도 대상에 포함한다(only_missing=False = --all 스캔).
        conn, cur = _mock_conn(fetchall_val=[("a1",), ("a2",)])
        _fetch_target_asset_ids(conn, only_missing=False)
        sql = cur.execute.call_args.args[0].lower()
        self.assertNotIn("left join asset_topic", sql)  # IS NULL 미부여 필터 없음
        self.assertIn("status = 'registered'", sql)

    def test_none_result_deletes_existing_row_and_counts(self):
        # 재분류 결과 미부여(None) → delete_fn 으로 기존 행 삭제·삭제 카운트.
        def classify(conn, aid, *, settings=None, client=None):
            return None  # 재분류 결과 미부여(예: 미분류/무내용으로 전이)

        deleted_ids: list = []

        def delete_fn(conn, aid):
            deleted_ids.append(aid)
            return 1  # rowcount(실제 행 삭제됨)

        s = self._run(ids=["a1", "a2"], classify_fn=classify, delete_fn=delete_fn)
        self.assertEqual(deleted_ids, ["a1", "a2"])  # 두 자산 모두 삭제 시도
        self.assertEqual(s["deleted"], 2)
        self.assertEqual(s["classified"], 0)

    def test_none_result_no_existing_row_no_delete_count(self):
        # 재분류 None 인데 삭제할 행이 없으면(rowcount 0) deleted 카운트 증가 없음.
        def classify(conn, aid, *, settings=None, client=None):
            return None

        def delete_fn(conn, aid):
            return 0  # 삭제할 행 없음(원래 미부여)

        s = self._run(ids=["a1"], classify_fn=classify, delete_fn=delete_fn)
        self.assertEqual(s["deleted"], 0)

    def test_dict_result_upserts_no_delete(self):
        # 재분류 결과 부여(dict) → classify_fn 이 upsert(정본 갱신). delete 미호출.
        def classify(conn, aid, *, settings=None, client=None):
            return {"topic_ko": "과학"}

        delete_fn = MagicMock()
        s = self._run(ids=["a1"], classify_fn=classify, delete_fn=delete_fn)
        self.assertEqual(s["classified"], 1)
        self.assertEqual(s["deleted"], 0)
        delete_fn.assert_not_called()

    def test_default_mode_none_does_not_delete(self):
        # 기본(비-reclassify): None 은 미부여로 두고 삭제하지 않는다(기존 동작 보존).
        def classify(conn, aid, *, settings=None, client=None):
            return None

        delete_fn = MagicMock()
        s = self._run(
            ids=["a1"], classify_fn=classify, delete_fn=delete_fn, reclassify=False
        )
        self.assertEqual(s["no_text"], 1)
        self.assertEqual(s["deleted"], 0)
        delete_fn.assert_not_called()


# ────────────────────────────────────────────────────────────────────────────
# 3) 현황 리포트(--report) — 순수 집계
# ────────────────────────────────────────────────────────────────────────────
class TestStatusReport(unittest.TestCase):
    def test_assignment_rate_and_missing(self):
        rep = build_status_report(
            {"n_registered": 100, "n_with_meta": 90, "n_with_topic": 72}
        )
        self.assertEqual(rep["n_missing"], 18)  # 90 - 72
        self.assertEqual(rep["assignment_rate"], 0.8)  # 72 / 90

    def test_zero_meta_safe(self):
        rep = build_status_report(
            {"n_registered": 0, "n_with_meta": 0, "n_with_topic": 0}
        )
        self.assertEqual(rep["assignment_rate"], 0.0)
        self.assertEqual(rep["n_missing"], 0)

    def test_format_lines_contain_metrics(self):
        rep = build_status_report(
            {"n_registered": 10, "n_with_meta": 8, "n_with_topic": 6}
        )
        text = "\n".join(format_status_lines(rep))
        self.assertIn("부여율", text)
        self.assertIn("6", text)


if __name__ == "__main__":
    unittest.main()
