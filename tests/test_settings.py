"""021 G4 — 검색 백엔드 설정 3필드 정식화 + fail-fast 검증 (순수 단위 테스트).

020(OpenSearch 동기화)이 깐 ``opensearch_url``/``opensearch_index``/``opensearch_sync_enabled``
선택 필드 패턴(``_env_str_default``/``_env_bool_default`` + ``_build_settings`` 조립)을 따라,
021 검색 read path 전환(spec 021 FR-005·006·010)에 필요한 아래 3필드를 정식화한다.

  - ``search_backend``           : 기본 ``"pg"``, 화이트리스트 ``{"pg","opensearch"}`` — 그 외 값이면
                                   ``_build_settings``(=init_settings 검증 지점)에서 **즉시 ValueError**
                                   (런타임까지 숨지 않게 — 백로그 '설정 fail-late' 교정).
  - ``opensearch_search_pipeline``: 기본 ``"assets-hybrid"`` (문자열, normalization-processor 이름).
  - ``opensearch_fusion_weights`` : 기본 ``(0.5, 0.5)`` (BM25, kNN). 각 가중치 **0<=w<=1**(벗어나면 ValueError).
                                    ``OPENSEARCH_FUSION_WEIGHTS="0.5,0.5"`` → 튜플로 파싱.

⚠️ G3(``search_service.search_hybrid``)가 이 3필드를 ``getattr(cfg, "search_backend", "pg")`` /
``getattr(cfg, "opensearch_search_pipeline", "assets-hybrid")`` / ``getattr(cfg,
"opensearch_fusion_weights", (0.5,0.5))`` 로 읽으므로 **필드명·기본값이 정확히 일치**해야 한다 —
``TestG3FieldNameContract`` 가 그 계약을 봉인한다.

``_build_settings`` 는 11개 필수 env 를 요구하므로(test_settings_relation_retry 동형), 그 최소 env 를
임시로 채운 뒤 검색 백엔드 키만 토글한다(다른 테스트 환경을 오염시키지 않도록 정확히 원복).
"""

from __future__ import annotations

import contextlib
import os
import unittest
from unittest import mock

from src.config.settings import _build_settings

# _build_settings 가 _require_env* 로 읽는 필수 env 최소 집합(값은 형식만 맞으면 됨).
_REQUIRED_ENV = {
    "META_MODEL": "gemma",
    "OPENAI_BASE_URL": "http://localhost:1234/v1",
    "OPENAI_API_KEY": "sk-test",
    "SUMMARY_MAX_CHARS": "500",
    "TOP_K_KEYWORDS": "10",
    "CHUNK_SIZE": "1000",
    "OVERLAP_SIZE": "100",
    "ENCODING": "utf-8",
    "TEXT_EMBED_MODEL": "bge-m3",
    "TEXT_EMBED_CHUNK_SIZE": "512",
    "TEXT_EMBED_NORMALIZE": "true",
}

# 021 G4 가 추가하는 검색 백엔드 선택 env 키(테스트 격리를 위해 매 케이스 깨끗이 비운다).
_BACKEND_KEYS = (
    "SEARCH_BACKEND",
    "OPENSEARCH_SEARCH_PIPELINE",
    "OPENSEARCH_FUSION_WEIGHTS",
)


@contextlib.contextmanager
def _env(**overrides: str):
    """필수 env 를 임시로 채우고 검색 백엔드 키를 비운 뒤 ``overrides`` 만 설정·복원한다."""
    touched = list(_REQUIRED_ENV) + list(_BACKEND_KEYS) + list(overrides)
    saved = {k: os.environ.get(k) for k in touched}
    try:
        os.environ.update(_REQUIRED_ENV)
        for k in _BACKEND_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: str(v) for k, v in overrides.items()})
        yield
    finally:
        for k in touched:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


class TestSearchBackend(unittest.TestCase):
    """``search_backend``: 기본 'pg' · 화이트리스트 fail-fast(FR-010·plan §3)."""

    def test_default_is_pg_when_unset(self) -> None:
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.search_backend, "pg")

    def test_env_override_opensearch(self) -> None:
        with _env(SEARCH_BACKEND="opensearch"):
            settings = _build_settings("dev")
        self.assertEqual(settings.search_backend, "opensearch")

    def test_invalid_backend_raises_fail_fast(self) -> None:
        # 화이트리스트 밖 값은 init_settings(=_build_settings) 에서 즉시 ValueError — 런타임까지 숨지 않게.
        with _env(SEARCH_BACKEND="elastic"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_invalid_backend_empty_after_strip_uses_default(self) -> None:
        # 공백만 있으면 _env_str_default 관례상 미설정 취급 → 기본 'pg'(빈 문자열로 검증 실패시키지 않음).
        with _env(SEARCH_BACKEND="   "):
            settings = _build_settings("dev")
        self.assertEqual(settings.search_backend, "pg")


class TestOpenSearchSearchPipeline(unittest.TestCase):
    """``opensearch_search_pipeline``: 기본 'assets-hybrid'(문자열) — _env_str_default 패턴."""

    def test_default_is_assets_hybrid(self) -> None:
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.opensearch_search_pipeline, "assets-hybrid")

    def test_env_override(self) -> None:
        with _env(OPENSEARCH_SEARCH_PIPELINE="custom-pipe"):
            settings = _build_settings("dev")
        self.assertEqual(settings.opensearch_search_pipeline, "custom-pipe")


class TestOpenSearchFusionWeights(unittest.TestCase):
    """``opensearch_fusion_weights``: 기본 (0.5,0.5) · 'w1,w2' 파싱 · 0<=w<=1 범위검증(FR-005)."""

    def test_default_is_half_half(self) -> None:
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.opensearch_fusion_weights, (0.5, 0.5))

    def test_env_override_parses_tuple(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="0.3,0.7"):
            settings = _build_settings("dev")
        self.assertEqual(settings.opensearch_fusion_weights, (0.3, 0.7))

    def test_whitespace_around_values_tolerated(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS=" 0.4 , 0.6 "):
            settings = _build_settings("dev")
        self.assertEqual(settings.opensearch_fusion_weights, (0.4, 0.6))

    def test_boundary_values_zero_one_ok(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="0,1"):
            settings = _build_settings("dev")
        self.assertEqual(settings.opensearch_fusion_weights, (0.0, 1.0))

    def test_above_one_raises(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="1.5,0.5"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_negative_raises(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="-0.1,0.5"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_wrong_count_too_few_raises(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="0.5"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_wrong_count_too_many_raises(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="0.3,0.3,0.4"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_non_numeric_raises(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="a,b"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestG3FieldNameContract(unittest.TestCase):
    """G3(search_service.search_hybrid) getattr 계약 봉인 — 필드명·기본값 정확 일치.

    G3 가 ``getattr(cfg, "search_backend", "pg")`` / ``getattr(cfg, "opensearch_search_pipeline",
    "assets-hybrid")`` / ``getattr(cfg, "opensearch_fusion_weights", (0.5,0.5))`` 로 읽으므로,
    정식화된 필드명·기본값이 그 폴백값과 어긋나면 동작이 갈라진다(회귀). 이 계약을 직접 봉인한다.
    """

    def test_field_names_and_defaults_match_g3_getattr(self) -> None:
        with _env():
            settings = _build_settings("dev")
        # 필드명(getattr 키) 일치
        self.assertTrue(hasattr(settings, "search_backend"))
        self.assertTrue(hasattr(settings, "opensearch_search_pipeline"))
        self.assertTrue(hasattr(settings, "opensearch_fusion_weights"))
        # 기본값이 G3 getattr 폴백과 동일(미설정 시 동작 불변)
        self.assertEqual(settings.search_backend, "pg")
        self.assertEqual(settings.opensearch_search_pipeline, "assets-hybrid")
        self.assertEqual(settings.opensearch_fusion_weights, (0.5, 0.5))


class TestSearchBackendWiring(unittest.TestCase):
    """T008 스모크: ``SEARCH_BACKEND`` 설정이 ``search_hybrid`` 백엔드 경로에 반영된다.

    진입점(run_search·portal_api·sample_search_api)은 ``backend`` 인자 없이 ``search_hybrid`` 를
    호출하므로, 백엔드 선택은 전적으로 ``settings.search_backend`` 가 제어한다 — 즉 진입점 **호출부
    코드 변경이 불필요**함을 봉인한다(plan §4). settings 전역을 오염시키지 않도록 ``get_current_settings``
    를 모킹해 빌드된 설정을 주입한다.
    """

    def test_opensearch_setting_routes_search_hybrid_to_os(self) -> None:
        import src.search.search_service as svc

        with _env(SEARCH_BACKEND="opensearch"):
            cfg = _build_settings("dev")

        os_cap: dict[str, object] = {}

        def fake_os(client: object, query: str, **kw: object) -> dict[str, list[dict[str, object]]]:
            os_cap["client"] = client
            os_cap["query"] = query
            os_cap.update(kw)
            return {"text": [{"id": "os_t"}]}

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            raise AssertionError("opensearch 백엔드 text 버킷에 pg grouped 가 쓰이면 안 됨")

        # backend 인자 미전달(진입점 호출부와 동일) → settings.search_backend 가 경로를 결정한다.
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "FAKE",
                _grouped_fn=fake_grouped,
            )
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t"}])
        self.assertEqual(os_cap["client"], "FAKE")

    def test_default_pg_setting_routes_search_hybrid_to_pg(self) -> None:
        import src.search.search_service as svc

        with _env():  # SEARCH_BACKEND 미설정 → 기본 'pg'
            cfg = _build_settings("dev")

        os_calls: list[object] = []

        def fake_os(*a: object, **k: object) -> dict[str, list[dict[str, object]]]:
            os_calls.append(1)
            return {}

        def fake_grouped(query: str, **kw: object) -> dict[str, object]:
            return {"text_documents": [{"id": "pg_t"}], "meta": {}}

        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "FAKE",
                _grouped_fn=fake_grouped,
            )
        self.assertEqual(os_calls, [])  # 기본 pg → OS seam 미접촉(회귀 0·SC-001)
        self.assertEqual(out["results"]["text_documents"], [{"id": "pg_t"}])


if __name__ == "__main__":
    unittest.main()
