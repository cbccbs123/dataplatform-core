"""v2 단계 B — run_ingest 팩 경로(override 없음) + 정책 검증 테스트(목 영속화)."""
from __future__ import annotations

import contextlib
import unittest
import uuid
from unittest import mock

from src.app import run_ingest as ri
from src.classify.types import ClassificationResult
from src.dispatch.types import AssetRecord, EmbeddingItem
from src.ingest.router import RouteResult
from src.pipeline.registry import StrategyRegistry


def _route(modality="txt", domain="general"):
    return RouteResult("/d/a." + modality, modality, domain, True, "")


def _fake_registry(captured):
    r = StrategyRegistry()
    r.register("classify", "cascade_v1", lambda ctx: ClassificationResult(final_label="general", confidence=0.7, decided_stage=2))

    def _extract(ctx):
        return AssetRecord(fts_plain="x")

    def _embed(ctx, rec):
        captured["embed_called"] = True
        return [EmbeddingItem(channel="st", vector=[0.1], model_name="m")]

    r.register("extract", "by_modality", _extract)
    r.register("embed", "by_modality", _embed)
    r.register("persist", "asset_upsert", lambda *a: None)
    return r


class TestRunIngestPackPath(unittest.TestCase):
    def _patch(self, stack):
        stack.enter_context(mock.patch.object(ri, "route_file", return_value=_route()))
        stack.enter_context(mock.patch.object(ri, "file_hash_and_size", return_value=("h", 1)))
        stack.enter_context(mock.patch.object(ri, "find_registered_asset_by_hash", return_value=None))
        stack.enter_context(mock.patch.object(ri, "create_asset", return_value=uuid.UUID(int=1)))
        stack.enter_context(mock.patch.object(ri, "record_classification"))
        stack.enter_context(mock.patch.object(ri, "set_status"))
        stack.enter_context(mock.patch.object(ri, "finalize_asset"))

    def test_pack_path_runs_extract_embed_and_validates_policy(self) -> None:
        captured: dict = {}
        reg = _fake_registry(captured)
        with contextlib.ExitStack() as stack:
            self._patch(stack)
            with mock.patch.object(ri, "policy_validate") as mval:
                res = ri.run_ingest(["/d/a.txt"], db=mock.MagicMock(), registry=reg, settings=object())
        self.assertEqual(len(res["registered"]), 1)
        self.assertTrue(captured["embed_called"])      # embed 슬롯 호출됨(분리 경로)
        mval.assert_called()                            # 정책 검증 호출됨

    def test_override_extract_fn_skips_embed_slot(self) -> None:
        captured: dict = {}
        reg = _fake_registry(captured)
        with contextlib.ExitStack() as stack:
            self._patch(stack)
            ri.run_ingest(["/d/a.txt"], db=mock.MagicMock(), registry=reg,
                          extract_fn=lambda ctx: AssetRecord(fts_plain="ov"),
                          classify_fn=lambda p, m: ClassificationResult(final_label="general", confidence=0.7, decided_stage=2),
                          settings=object())
        self.assertNotIn("embed_called", captured)      # override 경로는 embed 슬롯 미호출


if __name__ == "__main__":
    unittest.main()
