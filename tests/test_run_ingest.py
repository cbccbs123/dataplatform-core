"""F-1.x run_ingest 오케스트레이션 단위 테스트 (모델 A + F-5.1 분류 통합).

persist/status/route/해시/분류 함수는 mock 으로 대체하고, 오케스트레이터의 제어 흐름
(성공/실패 흡수/skip/중복/분류 영속화/도메인 주입)만 검증한다. 실 DB·LLM·파일 불필요.
"""

from __future__ import annotations

import contextlib
import unittest
import uuid
from unittest import mock

from src.app import run_ingest as ri
from src.classify.types import ClassificationResult
from src.dispatch.dispatcher import UnsupportedModalityError
from src.dispatch.types import AssetRecord
from src.ingest.router import RouteResult

_EXISTING = uuid.UUID("018f0000-0000-7000-8000-000000000018")


def _route(modality="txt", routable=True, reason="", domain="general"):
    return RouteResult("/d/a." + modality, modality, domain, routable, reason)


def _cls(label="general", stage=2, conf=0.7):
    return ClassificationResult(final_label=label, confidence=conf, decided_stage=stage)


def _patch_all(stack: contextlib.ExitStack, route_result, *, dup=None) -> dict:
    """run_ingest 의존 함수 전부 patch 후 mock dict 반환."""
    specs = {
        "route_file": mock.patch.object(ri, "route_file", return_value=route_result),
        "file_hash_and_size": mock.patch.object(ri, "file_hash_and_size", return_value=("h0", 10)),
        "find_registered_asset_by_hash": mock.patch.object(ri, "find_registered_asset_by_hash", return_value=dup),
        "create_asset": mock.patch.object(ri, "create_asset", return_value=1),
        "record_classification": mock.patch.object(ri, "record_classification"),
        "set_status": mock.patch.object(ri, "set_status"),
        "finalize_asset": mock.patch.object(ri, "finalize_asset"),
        "mark_failed": mock.patch.object(ri, "mark_failed"),
        "record_lineage": mock.patch.object(ri, "record_lineage"),
        "validate_ext_meta": mock.patch.object(ri, "validate_ext_meta"),
    }
    return {k: stack.enter_context(p) for k, p in specs.items()}


class TestRunIngest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = object()
        self.db = mock.MagicMock()

    def _ingest(self, files, m_route, *, extract_fn, classify=None, dup=None, configure=None):
        with contextlib.ExitStack() as stack:
            m = _patch_all(stack, m_route, dup=dup)
            if configure:
                configure(m)
            res = ri.run_ingest(
                files, db=self.db, extract_fn=extract_fn,
                classify_fn=lambda p, mod: (classify or _cls()), settings=self.settings,
            )
        return res, m

    def test_success_path(self) -> None:
        res, m = self._ingest(["/d/a.txt"], _route(), extract_fn=lambda ctx: AssetRecord(fts_plain="x"))
        self.assertEqual(res["registered"], [1])
        m["create_asset"].assert_called_once()
        m["record_classification"].assert_called_once()
        m["finalize_asset"].assert_called_once()
        m["mark_failed"].assert_not_called()
        self.assertEqual(m["set_status"].call_count, 3)  # routing→classifying→extracting

    def test_missing_file_skipped(self) -> None:
        res, m = self._ingest(
            ["/no/x.txt"], _route(routable=False, reason=ri.REASON_MISSING), extract_fn=lambda ctx: AssetRecord()
        )
        self.assertEqual(len(res["skipped"]), 1)
        m["create_asset"].assert_not_called()
        m["record_classification"].assert_not_called()

    def test_duplicate_skipped(self) -> None:
        res, m = self._ingest(["/d/a.txt"], _route(), extract_fn=lambda ctx: AssetRecord(), dup=_EXISTING)
        self.assertEqual(res["registered"], [])
        self.assertEqual(len(res["skipped"]), 1)
        self.assertTrue(res["skipped"][0][1].startswith(ri.REASON_DUPLICATE))
        m["create_asset"].assert_not_called()

    def test_deferred_duplicate_skipped_no_new_row(self) -> None:
        # 009(#4): 이미 deferred 로 보류된 동일 해시(DICOM 재수집)는 find_registered_asset_by_hash
        # 가 기존 asset_id 를 dup 으로 돌려준다 → skip, 새 deferred/asset 행 미생성(중복 0).
        # run_ingest 의 skip 분기는 dup is not None 만 보므로 dedup 확장과 동일하게 동작한다.
        res, m = self._ingest(["/d/scan.dcm"], _route(modality="unknown"),
                              extract_fn=lambda ctx: AssetRecord(), dup=_EXISTING)
        self.assertEqual(res["registered"], [])
        self.assertEqual(res["deferred"], [])
        self.assertEqual(len(res["skipped"]), 1)
        self.assertTrue(res["skipped"][0][1].startswith(ri.REASON_DUPLICATE))
        m["create_asset"].assert_not_called()
        m["record_classification"].assert_not_called()

    def test_medical_standard_format_deferred(self) -> None:
        # DICOM 등 stage1 시그니처 → 추출 보류(deferred), 실패 아님, 추출 미호출.
        called = {"extract": False}

        def extract(ctx):
            called["extract"] = True
            return AssetRecord()

        cls = ClassificationResult(
            final_label="medical", confidence=1.0, decided_stage=1,
            stage1_scores={"medical": {"signature": "dicom"}},
        )
        res, m = self._ingest(
            ["/d/scan.dcm"], _route(modality="unknown", routable=False, reason="unknown_modality"),
            extract_fn=extract, classify=cls,
        )
        self.assertEqual(res["deferred"], [1])
        self.assertEqual(res["registered"], [])
        self.assertFalse(called["extract"])
        m["finalize_asset"].assert_not_called()
        m["mark_failed"].assert_not_called()
        m["record_classification"].assert_called_once()

    def test_classification_medical_sets_domain(self) -> None:
        seen = {}

        def capture(ctx):
            seen["domain"] = ctx.domain
            return AssetRecord()

        _, m = self._ingest(["/d/a.txt"], _route(), extract_fn=capture, classify=_cls(label="medical", stage=1, conf=1.0))
        self.assertEqual(seen["domain"], "medical")
        m["record_classification"].assert_called_once()

    def test_extract_exception_absorbed_as_failed(self) -> None:
        def boom(ctx):
            raise RuntimeError("추출 실패")
        res, m = self._ingest(["/d/a.txt"], _route(), extract_fn=boom)
        self.assertEqual(len(res["failed"]), 1)
        self.assertEqual(res["failed"][0][0], 1)
        m["finalize_asset"].assert_not_called()
        m["mark_failed"].assert_called_once()

    def test_unknown_modality_not_skipped_but_failed(self) -> None:
        def unsupported(ctx):
            raise UnsupportedModalityError(ctx.modality)
        res, m = self._ingest(
            ["/d/a.bin"], _route(modality="unknown", routable=False, reason="unknown_modality"), extract_fn=unsupported
        )
        m["create_asset"].assert_called_once()
        m["mark_failed"].assert_called_once()
        self.assertEqual(len(res["failed"]), 1)

    def test_create_asset_failure_isolated(self) -> None:
        res, m = self._ingest(
            ["/d/a.txt"], _route(), extract_fn=lambda ctx: AssetRecord(),
            configure=lambda m: setattr(m["create_asset"], "side_effect", RuntimeError("DB down")),
        )
        self.assertEqual(len(res["failed"]), 1)
        self.assertIsNone(res["failed"][0][0])
        m["finalize_asset"].assert_not_called()
        m["mark_failed"].assert_not_called()

    def test_batch_continues_after_one_failure(self) -> None:
        calls = {"n": 0}

        def extract(ctx):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("첫 파일 실패")
            return AssetRecord()

        res, m = self._ingest(
            ["/d/a.txt", "/d/b.txt"], _route(), extract_fn=extract,
            configure=lambda m: setattr(m["create_asset"], "side_effect", [1, 2]),
        )
        self.assertEqual(res["registered"], [2])
        self.assertEqual(len(res["failed"]), 1)
        self.assertEqual(res["failed"][0][0], 1)


if __name__ == "__main__":
    unittest.main()
