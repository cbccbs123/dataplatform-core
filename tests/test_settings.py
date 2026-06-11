"""021 G4 — 검색 백엔드 설정 3필드 정식화 + fail-fast 검증 (순수 단위 테스트).

020(OpenSearch 동기화)이 깐 ``opensearch_url``/``opensearch_index``/``opensearch_sync_enabled``
선택 필드 패턴(``_env_str_default``/``_env_bool_default`` + ``_build_settings`` 조립)을 따라,
021 검색 read path 전환(spec 021 FR-005·006·010)에 필요한 아래 3필드를 정식화한다.

  - ``search_backend``           : 기본 ``"pg"``, 화이트리스트 ``{"pg","opensearch"}`` — 그 외 값이면
                                   ``_build_settings``(=init_settings 검증 지점)에서 **즉시 ValueError**
                                   (런타임까지 숨지 않게 — 백로그 '설정 fail-late' 교정).
  - ``opensearch_fusion_weights`` : 기본 ``search_constants.OS_FUSION_WEIGHTS_DEFAULT``(=(0.5,0.5)).
                                    각 가중치 **0<=w<=1**(벗어나면 ValueError). ``"0.5,0.5"`` → 튜플로 파싱.

027 갱신: 서버 정규화 융합 파이프라인이 클라이언트 융합으로 대체되며 파이프라인 메타 필드·env 키가
제거됐다. 임계 기본값은 ``search_constants`` 단일 출처(F1)를 참조한다.

⚠️ G3(``search_service.search_hybrid``)가 이 필드들을 ``getattr(cfg, "search_backend", "pg")`` /
``getattr(cfg, "opensearch_fusion_weights", …)`` 로 읽으므로 **필드명·기본값이 정확히 일치**해야 한다 —
``TestG3FieldNameContract`` 가 그 계약을 봉인한다.

``_build_settings`` 는 11개 필수 env 를 요구하므로(test_settings_relation_retry 동형), 그 최소 env 를
임시로 채운 뒤 검색 백엔드 키만 토글한다(다른 테스트 환경을 오염시키지 않도록 정확히 원복).
"""

from __future__ import annotations

import contextlib
import os
import unittest
from unittest import mock

from src.config import search_constants
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
# 023 G3 가 추가하는 OS 검색 컷오프(probe 게이트) 선택 env 키도 함께 비워, 실행 환경에 남은 값이
# 기본값 단언을 오염시키지 않게 한다(021 백엔드 키와 동형 격리).
_BACKEND_KEYS = (
    "SEARCH_BACKEND",
    "OPENSEARCH_FUSION_WEIGHTS",
    "SEARCH_OS_CUTOFF_ENABLED",
    "SEARCH_OS_CUTOFF_EPS",
    "SEARCH_OS_CUTOFF_FLOOR",
    # 027: per-result 컷 코사인 하한(probe_k·정규화 스케일 4종을 대체). 잔존값 오염 방지로 비운다.
    "SEARCH_OS_RESULT_FLOOR",
    # 026 T006/T004 가 추가하는 색인 빌더 선택 env 키도 함께 비워, 실행 환경 잔존값이 기본값 단언을
    # 오염시키지 않게 한다(021 백엔드 키와 동형 격리).
    "OPENSEARCH_NORI_USER_WORDS",
    "OPENSEARCH_FILENAME_NOISE_PATTERNS",
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


class TestSearchOsCutoffSettings(unittest.TestCase):
    """023/027 — OS 검색 버킷 게이트(robust baseline) 3종 정식화 + fail-fast + search_constants 단일 출처.

    021 ``_resolve_opensearch_fusion_weights`` 패턴 미러: dataclass 필드 + ``_resolve_*`` 환경 파싱·
    범위 검증 + ``_build_settings`` 배선. 범위 밖 값은 init_settings(=_build_settings) 에서 **즉시
    ValueError** — 잘못된 임계로 검색하지 않게(fail-fast, _resolve_search_backend 동형).

      - ``search_os_cutoff_enabled`` : 기본 OS_CUTOFF_ENABLED_DEFAULT(True). pg 백엔드(기본)엔 무영향.
      - ``search_os_cutoff_eps``     : 기본 OS_CUTOFF_EPS_DEFAULT, 범위 ``[0,1)`` (상대신호 하한).
      - ``search_os_cutoff_floor``   : 기본 OS_CUTOFF_FLOOR_DEFAULT, 범위 ``[-1,1]`` (코사인 절대 backstop).

    027: 게이트 표본 수(search_os_probe_k)는 클라이언트 융합 전환으로 제거됐다(같은 kNN 표본에서 직접
    신호를 잰다 — 추가 검색 0). 기본값은 하드코딩 대신 search_constants 단일 출처를 참조한다(F1).
    """

    # (a) 기본값 — env 미설정 시 search_constants 단일 출처(F1) 값.
    def test_defaults_when_unset(self) -> None:
        with _env():
            settings = _build_settings("dev")
        self.assertIs(settings.search_os_cutoff_enabled, search_constants.OS_CUTOFF_ENABLED_DEFAULT)
        self.assertEqual(settings.search_os_cutoff_eps, search_constants.OS_CUTOFF_EPS_DEFAULT)
        self.assertEqual(settings.search_os_cutoff_floor, search_constants.OS_CUTOFF_FLOOR_DEFAULT)

    # (b) 환경 오버라이드.
    def test_enabled_env_override_false(self) -> None:
        with _env(SEARCH_OS_CUTOFF_ENABLED="false"):
            settings = _build_settings("dev")
        self.assertIs(settings.search_os_cutoff_enabled, False)

    def test_eps_env_override(self) -> None:
        with _env(SEARCH_OS_CUTOFF_EPS="0.2"):
            settings = _build_settings("dev")
        self.assertEqual(settings.search_os_cutoff_eps, 0.2)

    def test_floor_env_override(self) -> None:
        with _env(SEARCH_OS_CUTOFF_FLOOR="0.7"):
            settings = _build_settings("dev")
        self.assertEqual(settings.search_os_cutoff_floor, 0.7)

    # 경계 포함/제외 — eps∈[0,1)·floor∈[-1,1].
    def test_eps_boundary_zero_ok_one_excluded(self) -> None:
        with _env(SEARCH_OS_CUTOFF_EPS="0"):
            self.assertEqual(_build_settings("dev").search_os_cutoff_eps, 0.0)
        with _env(SEARCH_OS_CUTOFF_EPS="1"):  # 1.0 은 범위 밖([0,1))
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_floor_boundary_inclusive(self) -> None:
        for v in ("-1", "1"):
            with _env(SEARCH_OS_CUTOFF_FLOOR=v):
                _build_settings("dev")  # [-1,1] 경계 포함 — 예외 없음

    # (c) fail-fast — 범위 밖은 즉시 ValueError.
    def test_eps_negative_raises(self) -> None:
        with _env(SEARCH_OS_CUTOFF_EPS="-0.1"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_floor_above_one_raises(self) -> None:
        with _env(SEARCH_OS_CUTOFF_FLOOR="2"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_floor_below_minus_one_raises(self) -> None:
        with _env(SEARCH_OS_CUTOFF_FLOOR="-1.5"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_eps_non_numeric_raises(self) -> None:
        with _env(SEARCH_OS_CUTOFF_EPS="abc"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestSearchOsResultFloor(unittest.TestCase):
    """027: OS per-result 컷 코사인 하한 ``search_os_result_floor`` — 024 정규화 스케일 4종을 대체.

    행 유지 = BM25 매칭(어휘 증거) OR 원시 코사인 ≥ 이 값(의미 증거). 단일 코사인 스케일이라 모달리티별
    분리가 불필요한 전역 1개. 기본값은 search_constants.OS_RESULT_FLOOR_DEFAULT 단일 출처(F1). 코사인
    정의역이라 범위 ``[-1,1]``, 밖이면 _build_settings 시점에 즉시 ValueError(fail-fast).
    """

    def test_default_when_unset(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertEqual(s.search_os_result_floor, search_constants.OS_RESULT_FLOOR_DEFAULT)

    def test_env_override(self) -> None:
        with _env(SEARCH_OS_RESULT_FLOOR="0.55"):
            s = _build_settings("dev")
        self.assertEqual(s.search_os_result_floor, 0.55)

    def test_boundary_inclusive(self) -> None:
        for v in ("-1", "1"):
            with _env(SEARCH_OS_RESULT_FLOOR=v):
                _build_settings("dev")  # [-1,1] 경계 포함 — 예외 없음

    def test_above_one_raises(self) -> None:
        with _env(SEARCH_OS_RESULT_FLOOR="1.5"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_below_minus_one_raises(self) -> None:
        with _env(SEARCH_OS_RESULT_FLOOR="-2"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_non_numeric_raises(self) -> None:
        with _env(SEARCH_OS_RESULT_FLOOR="abc"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestSearchOsBm25Operator(unittest.TestCase):
    """025 G1: BM25 operator 설정 — 기본 'or'(현행·회귀 0), 'and'=전토큰 매칭(F2). 화이트리스트 fail-fast."""

    def test_default_or(self) -> None:
        with _env():
            self.assertEqual(_build_settings("dev").search_os_bm25_operator, "or")

    def test_override_and(self) -> None:
        with _env(SEARCH_OS_BM25_OPERATOR="and"):
            self.assertEqual(_build_settings("dev").search_os_bm25_operator, "and")

    def test_whitelist_fail_fast(self) -> None:
        with _env(SEARCH_OS_BM25_OPERATOR="xor"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestOpenSearchNoriUserWords(unittest.TestCase):
    """026 T006(FR-004): nori user_dictionary 외래어 목록 — 기본 7종·CSV 오버라이드·빈 항목 fail-fast.

    인덱스 빌더가 분해 방지용으로 쓰는 단일 출처. 공백 항목은 nori user_dictionary 규칙으로 무의미·
    거부 대상이라 즉시 ValueError(fail-fast — _resolve_search_backend 동형).
    """

    def test_default_when_unset(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertEqual(
            s.opensearch_nori_user_words,
            ("아이패드", "아이폰", "스마트워치", "맥세이프", "에어팟", "갤럭시", "애플워치"),
        )

    def test_csv_override(self) -> None:
        with _env(OPENSEARCH_NORI_USER_WORDS="갤럭시탭, 버즈 ,에어팟맥스"):
            s = _build_settings("dev")
        self.assertEqual(s.opensearch_nori_user_words, ("갤럭시탭", "버즈", "에어팟맥스"))

    def test_blank_entry_fail_fast(self) -> None:
        with _env(OPENSEARCH_NORI_USER_WORDS="아이폰,,갤럭시"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_default_matches_build_index_body_default(self) -> None:
        # 단일 출처 계약: settings 기본 = opensearch_sync.build_index_body 기본 외래어 목록(드리프트 차단).
        from src.search.opensearch_sync import _DEFAULT_NORI_USER_WORDS

        with _env():
            s = _build_settings("dev")
        self.assertEqual(tuple(s.opensearch_nori_user_words), tuple(_DEFAULT_NORI_USER_WORDS))


class TestOpenSearchFilenameNoisePatterns(unittest.TestCase):
    """026 T004(FR-003③): 파일명 정제 추가 잡음 패턴(regex) 목록 — 기본 빈 목록·CSV·컴파일 검증."""

    def test_default_empty(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertEqual(s.opensearch_filename_noise_patterns, ())

    def test_csv_override(self) -> None:
        with _env(OPENSEARCH_FILENAME_NOISE_PATTERNS=r"^shorts$, ^clip\d+$"):
            s = _build_settings("dev")
        self.assertEqual(s.opensearch_filename_noise_patterns, (r"^shorts$", r"^clip\d+$"))

    def test_blank_entry_fail_fast(self) -> None:
        with _env(OPENSEARCH_FILENAME_NOISE_PATTERNS="^a$,,^b$"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_invalid_regex_fail_fast(self) -> None:
        # 컴파일 불가 패턴은 잘못된 정제로 색인하지 않도록 즉시 ValueError.
        with _env(OPENSEARCH_FILENAME_NOISE_PATTERNS="[unterminated"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestG3FieldNameContract(unittest.TestCase):
    """G3(search_service.search_hybrid) getattr 계약 봉인 — 필드명·기본값 정확 일치.

    G3 가 ``getattr(cfg, "search_backend", "pg")`` / ``getattr(cfg, "opensearch_fusion_weights",
    search_constants.OS_FUSION_WEIGHTS_DEFAULT)`` / ``getattr(cfg, "search_os_result_floor",
    search_constants.OS_RESULT_FLOOR_DEFAULT)`` 로 읽으므로, 정식화된 필드명·기본값이 그 폴백값과
    어긋나면 동작이 갈라진다(회귀). 이 계약을 직접 봉인한다. 027: 서버 파이프라인 메타 필드는
    클라이언트 융합 전환으로 제거됐다 — 제거 봉인은 아래 '제거된 필드 부재' 단언이 담당한다.
    """

    # 027: 제거된 파이프라인 메타 필드명(부재를 봉인). 리터럴을 한 곳에만 두어 잔존 참조와 구분한다.
    _REMOVED_FIELDS = ("opensearch_" + "search_pipeline", "search_os_" + "probe_k", "search_os_" + "min_scores")

    def test_field_names_and_defaults_match_g3_getattr(self) -> None:
        with _env():
            settings = _build_settings("dev")
        # 필드명(getattr 키) 일치
        self.assertTrue(hasattr(settings, "search_backend"))
        self.assertTrue(hasattr(settings, "opensearch_fusion_weights"))
        self.assertTrue(hasattr(settings, "search_os_result_floor"))
        # 027: 제거된 필드들은 더 이상 존재하지 않는다(서버 융합 파이프라인·정규화 스케일 임계 0)
        for removed in self._REMOVED_FIELDS:
            self.assertFalse(hasattr(settings, removed), f"제거된 필드가 잔존: {removed}")
        # 기본값이 G3 getattr 폴백(search_constants 단일 출처)과 동일(미설정 시 동작 불변)
        self.assertEqual(settings.search_backend, "pg")
        self.assertEqual(settings.opensearch_fusion_weights, search_constants.OS_FUSION_WEIGHTS_DEFAULT)
        self.assertEqual(settings.search_os_result_floor, search_constants.OS_RESULT_FLOOR_DEFAULT)


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

        def fake_os(
            client: object, query: str, **kw: object
        ) -> tuple[dict[str, list[dict[str, object]]], dict]:
            os_cap["client"] = client
            os_cap["query"] = query
            os_cap.update(kw)
            # 027: search_assets_os 는 (buckets, gate_meta) 튜플을 돌려준다. 본 테스트는 라우팅 검증이
            # 목적이라 OS 경로가 호출부 필터를 적용하지 않으므로 행은 그대로 보존된다.
            return {"text": [{"id": "os_t", "similarity": 0.9}]}, {}

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
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t", "similarity": 0.9}])
        self.assertEqual(os_cap["client"], "FAKE")

    def test_default_pg_setting_routes_search_hybrid_to_pg(self) -> None:
        import src.search.search_service as svc

        with _env():  # SEARCH_BACKEND 미설정 → 기본 'pg'
            cfg = _build_settings("dev")

        os_calls: list[object] = []

        def fake_os(*a: object, **k: object) -> tuple[dict[str, list[dict[str, object]]], dict]:
            os_calls.append(1)
            return {}, {}

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
