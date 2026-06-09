"""run_ingest OpenSearch 증분 훅 단위 테스트 (spec 020 G3 — OS·DB 불필요).

이 그룹은 **프로덕션 적재(run_ingest) 경로 변경**이라 안전 게이트가 최우선이다:

  (a) ``OPENSEARCH_SYNC_ENABLED`` 미설정(기본) → 훅 헬퍼가 **즉시 반환**한다. opensearch_sync /
      opensearch-py 를 **import 도 하지 않는다**(지연 import). 따라서 미도입 환경의 run_ingest 가
      완전 불변(SC-001, 회귀 0). sys.modules 에 센티넬을 꽂아 그 모듈의 어떤 속성도 접근되지 않음을 단언.
  (b) enabled + 색인 실패(예외 주입) → 헬퍼가 예외를 **삼키고**(warning 로그) **정상 반환**한다.
      OS 가 죽어도 ingest 는 중단·롤백되지 않는다(SC-003, 격리).
  (c) enabled + 정상 → ``index_asset`` 이 해당 asset_id 로 호출된다(증분 색인 정상 경로, FR-002).

모든 검증은 **가짜 settings·db·opensearch_sync 모듈 주입**으로 OS·DB 없이 수행한다 — 실제 색인은 G5.
"""

from __future__ import annotations

import contextlib
import sys
import types
import unittest
import uuid
from unittest import mock

from src.app import run_ingest as ri

_ASSET = uuid.UUID("018f0000-0000-7000-8000-000000000019")


def _settings(*, enabled: bool, channel: str = "st") -> types.SimpleNamespace:
    """run_ingest 가 cfg 로 쓰는 설정 대역(필요 필드만)."""
    return types.SimpleNamespace(
        opensearch_sync_enabled=enabled,
        opensearch_url="http://localhost:9200",
        opensearch_index="assets",
        active_embed_channel=channel,
    )


class _RecordingModule(types.ModuleType):
    """sys.modules 에 꽂는 가짜 opensearch_sync 모듈 — 어떤 속성이든 접근되면 기록한다.

    훅이 ``from src.search.opensearch_sync import ...`` 를 실행하면(=import) 속성 접근이 일어나
    ``accessed`` 에 남는다. off 경로에서 이 리스트가 비어 있으면 **모듈을 import 조차 안 한 것**(SC-001).
    """

    def __init__(self) -> None:
        super().__init__("src.search.opensearch_sync")
        self.accessed: list[str] = []
        self.index_asset = mock.MagicMock(name="index_asset")
        self.get_client = mock.MagicMock(name="get_client", return_value=mock.MagicMock())

    def __getattribute__(self, name: str):
        # 내부 속성(accessed/index_asset/get_client/dunder)은 기록 대상에서 제외, 그 외 접근만 기록.
        if not name.startswith("_") and name not in {"accessed", "index_asset", "get_client"}:
            object.__getattribute__(self, "accessed").append(name)
        return object.__getattribute__(self, name)


@contextlib.contextmanager
def _patched_os_sync():
    """가짜 opensearch_sync 모듈을 sys.modules 에 주입(지연 import 가 이걸 잡는다)."""
    fake = _RecordingModule()
    with mock.patch.dict(sys.modules, {"src.search.opensearch_sync": fake}):
        yield fake


class TestOpenSearchHookHelper(unittest.TestCase):
    """격리 헬퍼 ``_opensearch_index_best_effort`` 단위 검증."""

    def setUp(self) -> None:
        self.db = mock.MagicMock()

    # (a) off → 즉시 반환 + opensearch_sync 미접근(미import)
    def test_disabled_returns_immediately_without_touching_module(self) -> None:
        with _patched_os_sync() as fake:
            ri._opensearch_index_best_effort(_ASSET, db=self.db, settings=_settings(enabled=False))
        # index/client 미호출
        fake.index_asset.assert_not_called()
        fake.get_client.assert_not_called()
        # 모듈 속성 접근이 전혀 없었음 = from ... import 자체가 실행되지 않음(지연 import 보장)
        self.assertEqual(fake.accessed, [])
        # DB 트랜잭션도 열지 않음(읽기 conn 미오픈)
        self.db.transaction.assert_not_called()

    # (a') 설정에 필드가 아예 없어도(레거시 settings) off 로 간주, 안전 반환(getattr 폴백)
    def test_missing_flag_attr_treated_as_off(self) -> None:
        legacy = object()  # opensearch_sync_enabled 속성 없음
        with _patched_os_sync() as fake:
            ri._opensearch_index_best_effort(_ASSET, db=self.db, settings=legacy)
        fake.index_asset.assert_not_called()
        self.assertEqual(fake.accessed, [])

    # (b) enabled + 색인 예외 → 삼킴(warning) + 무중단 반환
    def test_enabled_index_error_is_swallowed_with_warning(self) -> None:
        with _patched_os_sync() as fake:
            fake.index_asset.side_effect = RuntimeError("OS down")
            with self.assertLogs(ri._LOG, level="WARNING") as log:
                # 예외가 전파되면 이 호출이 raise → 테스트 실패. 격리 성공이면 정상 반환.
                ret = ri._opensearch_index_best_effort(
                    _ASSET, db=self.db, settings=_settings(enabled=True)
                )
        self.assertIsNone(ret)
        self.assertTrue(any("asset_id" in m or "opensearch" in m.lower() for m in log.output))

    # (b') enabled + get_client 단계 예외(OS 미도달)도 삼킨다 — 적재 무중단
    def test_enabled_client_error_is_swallowed(self) -> None:
        with _patched_os_sync() as fake:
            fake.get_client.side_effect = ConnectionError("unreachable")
            with self.assertLogs(ri._LOG, level="WARNING"):
                ret = ri._opensearch_index_best_effort(
                    _ASSET, db=self.db, settings=_settings(enabled=True)
                )
        self.assertIsNone(ret)
        fake.index_asset.assert_not_called()

    # (c) enabled + 정상 → index_asset 이 해당 asset_id 로 호출
    def test_enabled_indexes_with_asset_id(self) -> None:
        with _patched_os_sync() as fake:
            ri._opensearch_index_best_effort(_ASSET, db=self.db, settings=_settings(enabled=True))
        fake.index_asset.assert_called_once()
        call = fake.index_asset.call_args
        # asset_id 가 (문자열로) 전달되었는지 — 위치/키워드 어디에 있든 인자 집합에서 확인
        passed = list(call.args) + list(call.kwargs.values())
        self.assertIn(str(_ASSET), [str(a) for a in passed])
        # 색인 대상 채널/인덱스가 settings 에서 전달됨
        self.assertEqual(call.kwargs.get("index"), "assets")
        self.assertEqual(call.kwargs.get("channel"), "st")
        # 읽기 conn 을 열었음(PG 읽기전용 → OS 쓰기)
        self.db.transaction.assert_called_once()


# ── (a) run_ingest 전체 경로에서 off → OS 코드 미접촉(회귀 0) ──────────────────


def _route(modality="txt", routable=True, reason="", domain="general"):
    from src.ingest.router import RouteResult

    return RouteResult("/d/a." + modality, modality, domain, routable, reason)


def _patch_ingest(stack: contextlib.ExitStack) -> dict:
    """run_ingest 의존 함수 전부 patch(실 DB·LLM·파일 불필요) — test_run_ingest 와 동형."""
    from src.dispatch.types import AssetRecord

    specs = {
        "route_file": mock.patch.object(ri, "route_file", return_value=_route()),
        "file_hash_and_size": mock.patch.object(ri, "file_hash_and_size", return_value=("h0", 10)),
        "find_registered_asset_by_hash": mock.patch.object(
            ri, "find_registered_asset_by_hash", return_value=None
        ),
        "create_asset": mock.patch.object(ri, "create_asset", return_value=1),
        "record_classification": mock.patch.object(ri, "record_classification"),
        "set_status": mock.patch.object(ri, "set_status"),
        "finalize_asset": mock.patch.object(ri, "finalize_asset"),
        "mark_failed": mock.patch.object(ri, "mark_failed"),
        "record_lineage": mock.patch.object(ri, "record_lineage"),
        "validate_ext_meta": mock.patch.object(ri, "validate_ext_meta"),
    }
    m = {k: stack.enter_context(p) for k, p in specs.items()}
    m["_AssetRecord"] = AssetRecord
    return m


class TestRunIngestOffPathUnchanged(unittest.TestCase):
    """off(기본) 에서 run_ingest 가 OpenSearch 모듈을 전혀 건드리지 않음을 전체 경로로 단언."""

    def test_success_path_does_not_touch_opensearch_when_off(self) -> None:
        db = mock.MagicMock()
        with contextlib.ExitStack() as stack:
            m = _patch_ingest(stack)
            with _patched_os_sync() as fake:
                res = ri.run_ingest(
                    ["/d/a.txt"],
                    db=db,
                    extract_fn=lambda ctx: m["_AssetRecord"](fts_plain="x"),
                    classify_fn=lambda p, mod: __import__(
                        "src.classify.types", fromlist=["ClassificationResult"]
                    ).ClassificationResult(final_label="general", confidence=0.7, decided_stage=2),
                    settings=_settings(enabled=False),
                )
        # 적재 자체는 성공(기존 동작 불변)
        self.assertEqual(res["registered"], [1])
        # OpenSearch 모듈은 import 조차 안 됨(속성 접근 0 · 색인 0)
        self.assertEqual(fake.accessed, [])
        fake.index_asset.assert_not_called()
        fake.get_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
