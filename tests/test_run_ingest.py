"""F-1.x run_ingest 오케스트레이션 단위 테스트 (모델 A 흐름).

persist/status/route 함수는 이미 각자 단위 테스트가 있으므로 mock 으로 대체하고,
오케스트레이터의 **제어 흐름**(성공/실패 흡수/skip/단계 전이/도메인 주입)만 검증한다.
실 DB·LLM 불필요(db.transaction 은 MagicMock 컨텍스트 매니저).
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.app import run_ingest as ri
from src.dispatch.dispatcher import UnsupportedModalityError
from src.dispatch.types import AssetRecord
from src.ingest.router import RouteResult


def _route(modality="txt", routable=True, reason="", domain="general"):
    return RouteResult("/d/a." + modality, modality, domain, routable, reason)


def _patches(route_result):
    """run_ingest 가 쓰는 함수들을 patch. (route_file, create_asset, set_status, finalize_asset, mark_failed)"""
    return (
        mock.patch.object(ri, "route_file", return_value=route_result),
        mock.patch.object(ri, "create_asset", return_value=1),
        mock.patch.object(ri, "set_status"),
        mock.patch.object(ri, "finalize_asset"),
        mock.patch.object(ri, "mark_failed"),
    )


class TestRunIngest(unittest.TestCase):
    def setUp(self) -> None:
        # get_current_settings 호출 회피용 settings 주입
        self.settings = object()
        self.db = mock.MagicMock()

    def _run(self, files, route_result, *, extract_fn, classify_fn=None):
        p_route, p_create, p_set, p_final, p_fail = _patches(route_result)
        with p_route as route_file, p_create as create_asset, p_set as set_status, \
                p_final as finalize_asset, p_fail as mark_failed:
            res = ri.run_ingest(
                files, db=self.db, extract_fn=extract_fn, classify_fn=classify_fn,
                settings=self.settings,
            )
        return res, dict(
            route_file=route_file, create_asset=create_asset, set_status=set_status,
            finalize_asset=finalize_asset, mark_failed=mark_failed,
        )

    def test_success_path(self) -> None:
        rec = AssetRecord(fts_plain="x")
        res, m = self._run(["/d/a.txt"], _route(), extract_fn=lambda ctx: rec)
        self.assertEqual(res["registered"], [1])
        self.assertEqual(res["failed"], [])
        m["create_asset"].assert_called_once()
        m["finalize_asset"].assert_called_once()
        m["mark_failed"].assert_not_called()
        # routing→classifying→extracting = 3회 전이
        self.assertEqual(m["set_status"].call_count, 3)

    def test_missing_file_skipped(self) -> None:
        res, m = self._run(
            ["/no/x.txt"], _route(routable=False, reason=ri.REASON_MISSING),
            extract_fn=lambda ctx: AssetRecord(),
        )
        self.assertEqual(len(res["skipped"]), 1)
        m["create_asset"].assert_not_called()
        m["finalize_asset"].assert_not_called()

    def test_extract_exception_absorbed_as_failed(self) -> None:
        def boom(ctx):
            raise RuntimeError("추출 실패")
        res, m = self._run(["/d/a.txt"], _route(), extract_fn=boom)
        self.assertEqual(res["registered"], [])
        self.assertEqual(len(res["failed"]), 1)
        self.assertEqual(res["failed"][0][0], 1)
        m["finalize_asset"].assert_not_called()
        m["mark_failed"].assert_called_once()

    def test_unknown_modality_not_skipped_but_failed(self) -> None:
        # 디스패처 단일 권위: unknown 은 skip 이 아니라 생성 후 디스패처 예외→failed.
        def unsupported(ctx):
            raise UnsupportedModalityError(ctx.modality)
        res, m = self._run(
            ["/d/a.bin"], _route(modality="unknown", routable=False, reason="unknown_modality"),
            extract_fn=unsupported,
        )
        m["create_asset"].assert_called_once()
        m["mark_failed"].assert_called_once()
        self.assertEqual(len(res["failed"]), 1)

    def test_create_asset_failure_isolated(self) -> None:
        # asset 생성 단계 실패 → 배치 중단 없이 failed(None) 로 기록.
        p_route, p_create, p_set, p_final, p_fail = _patches(_route())
        with p_route, p_create as create_asset, p_set, p_final as finalize_asset, p_fail as mark_failed:
            create_asset.side_effect = RuntimeError("DB down")
            res = ri.run_ingest(["/d/a.txt"], db=self.db, extract_fn=lambda ctx: AssetRecord(), settings=self.settings)
        self.assertEqual(res["registered"], [])
        self.assertEqual(len(res["failed"]), 1)
        self.assertIsNone(res["failed"][0][0])  # asset_id 없음
        finalize_asset.assert_not_called()
        mark_failed.assert_not_called()  # asset 없으니 mark_failed 호출 안 함

    def test_batch_continues_after_one_failure(self) -> None:
        # 첫 파일 추출 실패해도 둘째 파일은 정상 등록(격리).
        calls = {"n": 0}

        def extract(ctx):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("첫 파일 실패")
            return AssetRecord()

        p_route, p_create, p_set, p_final, p_fail = _patches(_route())
        with p_route, p_create as create_asset, p_set, p_final, p_fail:
            create_asset.side_effect = [1, 2]
            res = ri.run_ingest(["/d/a.txt", "/d/b.txt"], db=self.db, extract_fn=extract, settings=self.settings)
        self.assertEqual(res["registered"], [2])
        self.assertEqual(len(res["failed"]), 1)
        self.assertEqual(res["failed"][0][0], 1)

    def test_classify_fn_sets_domain_on_context(self) -> None:
        seen = {}

        def capture(ctx):
            seen["domain"] = ctx.domain
            return AssetRecord()

        self._run(["/d/a.txt"], _route(), extract_fn=capture, classify_fn=lambda p, m: "medical")
        self.assertEqual(seen["domain"], "medical")


if __name__ == "__main__":
    unittest.main()
