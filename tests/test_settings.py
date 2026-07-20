"""021 G4 — 검색 백엔드 설정 3필드 정식화 + fail-fast 검증 (순수 단위 테스트).

020(OpenSearch 동기화)이 깐 ``opensearch_url``/``opensearch_index``/``opensearch_sync_enabled``
선택 필드 패턴(``_env_str_default``/``_env_bool_default`` + ``_build_settings`` 조립)을 따라,
021 검색 read path 전환(spec 021 FR-005·006·010)에 필요한 아래 3필드를 정식화한다.

  - ``search_backend``           : 037 OpenSearch 전용 정리 후 기본 ``"opensearch"``, 화이트리스트
                                   ``{"opensearch"}`` — 그 외 값(과거 ``"pg"`` 포함)이면
                                   ``_build_settings``(=init_settings 검증 지점)에서 **즉시 ValueError**
                                   (런타임까지 숨지 않게 — 백로그 '설정 fail-late' 교정).
  - ``opensearch_fusion_weights`` : 기본 ``search_constants.OS_FUSION_WEIGHTS_DEFAULT``(=(0.5,0.5)).
                                    각 가중치 **0<=w<=1**(벗어나면 ValueError). ``"0.5,0.5"`` → 튜플로 파싱.

027 갱신: 서버 정규화 융합 파이프라인이 클라이언트 융합으로 대체되며 파이프라인 메타 필드·env 키가
제거됐다. 임계 기본값은 ``search_constants`` 단일 출처(F1)를 참조한다.

⚠️ ``search_service.search_hybrid`` 가 OS 컷오프·융합 필드들을 ``getattr(cfg, "opensearch_fusion_weights",
…)`` 로 읽으므로 **필드명·기본값이 정확히 일치**해야 한다 — ``TestG3FieldNameContract`` 가 그 계약을
봉인한다. 037: search_backend 기본값은 'pg'→'opensearch' 로 전환됐다(PG 검색 경로 제거).

``_build_settings`` 는 11개 필수 env 를 요구하므로(test_settings_relation_retry 동형), 그 최소 env 를
임시로 채운 뒤 검색 백엔드 키만 토글한다(다른 테스트 환경을 오염시키지 않도록 정확히 원복).
"""

from __future__ import annotations

import contextlib
import os
import unittest
from unittest import mock

from src.config import search_constants
from src.config.settings import _FIELD_SPECS, _build_settings

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

# 069 US-E FR-E4(PR4a): 격리할 선택 env 키는 수동 나열 대신 settings 의 _FIELD_SPECS 단일 출처에서
# 파생한다(required=False 행의 env 키). 새 선택 필드 추가 시 이 목록이 자동 갱신 — 테스트 수정 0(SC-E).
_BACKEND_KEYS = tuple(_s.env for _s in _FIELD_SPECS if not _s.required)


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
    """``search_backend``: 037 OpenSearch 전용 정리 후 기본 'opensearch' · 화이트리스트 {'opensearch'}
    fail-fast(FR-010·037 plan §B4). 과거 'pg' 도 미지원 값이 됐다."""

    def test_default_is_opensearch_when_unset(self) -> None:
        # 037: PG 검색 경로 제거로 기본값이 'pg'→'opensearch' 로 전환됐다.
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.search.backend, "opensearch")

    def test_env_override_opensearch(self) -> None:
        with _env(SEARCH_BACKEND="opensearch"):
            settings = _build_settings("dev")
        self.assertEqual(settings.search.backend, "opensearch")

    def test_invalid_backend_raises_fail_fast(self) -> None:
        # 화이트리스트 밖 값은 init_settings(=_build_settings) 에서 즉시 ValueError — 런타임까지 숨지 않게.
        with _env(SEARCH_BACKEND="elastic"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_pg_backend_now_unsupported(self) -> None:
        # 037: 과거 기본값 'pg' 는 검색 경로 제거로 미지원 값이 됐다 → 즉시 ValueError(fail-fast).
        with _env(SEARCH_BACKEND="pg"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_invalid_backend_empty_after_strip_uses_default(self) -> None:
        # 공백만 있으면 _env_str_default 관례상 미설정 취급 → 기본 'opensearch'(빈 문자열로 검증 실패시키지 않음).
        with _env(SEARCH_BACKEND="   "):
            settings = _build_settings("dev")
        self.assertEqual(settings.search.backend, "opensearch")


class TestOpenSearchFusionWeights(unittest.TestCase):
    """``opensearch_fusion_weights``: 기본 (0.5,0.5) · 'w1,w2' 파싱 · 0<=w<=1 범위검증(FR-005)."""

    def test_default_is_half_half(self) -> None:
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.search.fusion_weights, (0.5, 0.5))

    def test_env_override_parses_tuple(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="0.3,0.7"):
            settings = _build_settings("dev")
        self.assertEqual(settings.search.fusion_weights, (0.3, 0.7))

    def test_whitespace_around_values_tolerated(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS=" 0.4 , 0.6 "):
            settings = _build_settings("dev")
        self.assertEqual(settings.search.fusion_weights, (0.4, 0.6))

    def test_boundary_values_zero_one_ok(self) -> None:
        with _env(OPENSEARCH_FUSION_WEIGHTS="0,1"):
            settings = _build_settings("dev")
        self.assertEqual(settings.search.fusion_weights, (0.0, 1.0))

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
        self.assertIs(settings.search.os_cutoff_enabled, search_constants.OS_CUTOFF_ENABLED_DEFAULT)
        self.assertEqual(settings.search.os_cutoff_eps, search_constants.OS_CUTOFF_EPS_DEFAULT)
        self.assertEqual(settings.search.os_cutoff_floor, search_constants.OS_CUTOFF_FLOOR_DEFAULT)

    # (b) 환경 오버라이드.
    def test_enabled_env_override_false(self) -> None:
        with _env(SEARCH_OS_CUTOFF_ENABLED="false"):
            settings = _build_settings("dev")
        self.assertIs(settings.search.os_cutoff_enabled, False)

    def test_eps_env_override(self) -> None:
        with _env(SEARCH_OS_CUTOFF_EPS="0.2"):
            settings = _build_settings("dev")
        self.assertEqual(settings.search.os_cutoff_eps, 0.2)

    def test_floor_env_override(self) -> None:
        with _env(SEARCH_OS_CUTOFF_FLOOR="0.7"):
            settings = _build_settings("dev")
        self.assertEqual(settings.search.os_cutoff_floor, 0.7)

    # 경계 포함/제외 — eps∈[0,1)·floor∈[-1,1].
    def test_eps_boundary_zero_ok_one_excluded(self) -> None:
        with _env(SEARCH_OS_CUTOFF_EPS="0"):
            self.assertEqual(_build_settings("dev").search.os_cutoff_eps, 0.0)
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


class TestSearchOsRerank(unittest.TestCase):
    """028: rerank 설정 4종 — 기본 off·fail-fast(τ∈[0,1]·R≥1)."""

    def test_defaults(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertIs(s.search.os_rerank_enabled, False)
        self.assertEqual(s.search.os_rerank_model, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(s.search.os_rerank_top_r, 10)

    def test_override_and_fail_fast(self) -> None:
        with _env(SEARCH_OS_RERANK_ENABLED="true", SEARCH_OS_RERANK_TAU="0.1"):
            s = _build_settings("dev")
        self.assertIs(s.search.os_rerank_enabled, True)
        self.assertEqual(s.search.os_rerank_tau, 0.1)
        with _env(SEARCH_OS_RERANK_TAU="1.5"):
            with self.assertRaises(ValueError):
                _build_settings("dev")
        with _env(SEARCH_OS_RERANK_TOP_R="0"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestSearchOsQueryNorm(unittest.TestCase):
    """029 T006: LLM 질의 명사구 정규화 토글 ``search_os_query_norm_enabled`` — 기본 off·env override.

    023 ``search_os_cutoff_enabled`` 의 bool resolver 패턴(``_env_bool_default``)을 미러한다 —
    범위검증 불필요한 순수 토글. 기본값은 search_constants.OS_QUERY_NORM_ENABLED_DEFAULT 단일 출처(F1).
    기본 off 라 미설정 환경·계약 테스트·027 폴백 불변(채택은 .env.dev opt-in)."""

    def test_default_off_when_unset(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertIs(
            s.search.os_query_norm_enabled, search_constants.OS_QUERY_NORM_ENABLED_DEFAULT
        )
        self.assertIs(s.search.os_query_norm_enabled, False)

    def test_env_override_true(self) -> None:
        with _env(SEARCH_OS_QUERY_NORM_ENABLED="true"):
            s = _build_settings("dev")
        self.assertIs(s.search.os_query_norm_enabled, True)

    def test_env_override_false_explicit(self) -> None:
        with _env(SEARCH_OS_QUERY_NORM_ENABLED="off"):
            s = _build_settings("dev")
        self.assertIs(s.search.os_query_norm_enabled, False)

    def test_invalid_bool_fail_fast(self) -> None:
        # 불리언 형식 오류는 _env_bool_default 가 즉시 ValueError(잘못된 토글로 검색하지 않게).
        with _env(SEARCH_OS_QUERY_NORM_ENABLED="maybe"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestSearchOsQueryNormMethod(unittest.TestCase):
    """075: 질의 정규화 방식 ``search_os_query_norm_method`` — 기본 morph·화이트리스트 fail-fast."""

    def test_default_is_morph(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertEqual(s.search.os_query_norm_method, search_constants.OS_QUERY_NORM_METHOD_DEFAULT)
        self.assertEqual(s.search.os_query_norm_method, "morph")

    def test_env_override_llm(self) -> None:
        with _env(SEARCH_OS_QUERY_NORM_METHOD="llm"):
            s = _build_settings("dev")
        self.assertEqual(s.search.os_query_norm_method, "llm")

    def test_case_insensitive(self) -> None:
        with _env(SEARCH_OS_QUERY_NORM_METHOD="LLM"):
            s = _build_settings("dev")
        self.assertEqual(s.search.os_query_norm_method, "llm")

    def test_invalid_method_fail_fast(self) -> None:
        # 화이트리스트 {morph, llm} 밖 값은 즉시 ValueError(_resolve_os_bm25_operator 동형).
        with _env(SEARCH_OS_QUERY_NORM_METHOD="gpt"):
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
        self.assertEqual(s.search.os_result_floor, search_constants.OS_RESULT_FLOOR_DEFAULT)

    def test_env_override(self) -> None:
        with _env(SEARCH_OS_RESULT_FLOOR="0.55"):
            s = _build_settings("dev")
        self.assertEqual(s.search.os_result_floor, 0.55)

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

    def test_default_and(self) -> None:
        # 027 리뷰 후속: 기본값 = 운영 검증값 'and'(F1 — 코드 기본=운영 보정값·어휘 구제 전제).
        with _env():
            self.assertEqual(_build_settings("dev").search.os_bm25_operator, "and")

    def test_override_and(self) -> None:
        with _env(SEARCH_OS_BM25_OPERATOR="and"):
            self.assertEqual(_build_settings("dev").search.os_bm25_operator, "and")

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
            s.opensearch.nori_user_words,
            ("아이패드", "아이폰", "스마트워치", "맥세이프", "에어팟", "갤럭시", "애플워치"),
        )

    def test_csv_override(self) -> None:
        with _env(OPENSEARCH_NORI_USER_WORDS="갤럭시탭, 버즈 ,에어팟맥스"):
            s = _build_settings("dev")
        self.assertEqual(s.opensearch.nori_user_words, ("갤럭시탭", "버즈", "에어팟맥스"))

    def test_blank_entry_fail_fast(self) -> None:
        with _env(OPENSEARCH_NORI_USER_WORDS="아이폰,,갤럭시"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_default_matches_build_index_body_default(self) -> None:
        # 069 T302(D6·P2-30): 단일 출처화 — settings 기본 = build_index_body 기본 = search_constants
        # 단일 상수(NORI_USER_WORDS_DEFAULT). 세 지점이 같은 값을 가리켜 드리프트가 원천 차단됨을 봉인
        # 한다(예전엔 settings·opensearch_sync 에 목록 2벌 + 이 테스트가 동치를 감시했다).
        from src.config import search_constants
        from src.search.opensearch_sync import build_index_body

        with _env():
            s = _build_settings("dev")
        rules = build_index_body()["settings"]["analysis"]["tokenizer"][
            "nori_user_tokenizer"
        ]["user_dictionary_rules"]
        self.assertEqual(
            tuple(s.opensearch.nori_user_words), search_constants.NORI_USER_WORDS_DEFAULT
        )
        self.assertEqual(tuple(rules), search_constants.NORI_USER_WORDS_DEFAULT)

    def test_default_defined_once_in_search_constants(self) -> None:
        # 069 T302: 기본 외래어 목록 리터럴이 src/ 전체에서 단 1곳(search_constants.py)에만.
        # 대표 항목 '맥세이프'(이 목록에만 등장)의 출현 파일로 단일 출처를 봉인한다(grep 계약).
        import pathlib

        src_root = pathlib.Path(__file__).resolve().parents[1] / "src"
        hits = sorted(
            p.name
            for p in src_root.rglob("*.py")
            if '"맥세이프"' in p.read_text(encoding="utf-8")
        )
        self.assertEqual(hits, ["search_constants.py"])


class TestOpenSearchFilenameNoisePatterns(unittest.TestCase):
    """026 T004(FR-003③): 파일명 정제 추가 잡음 패턴(regex) 목록 — 기본 빈 목록·CSV·컴파일 검증."""

    def test_default_empty(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertEqual(s.opensearch.filename_noise_patterns, ())

    def test_csv_override(self) -> None:
        with _env(OPENSEARCH_FILENAME_NOISE_PATTERNS=r"^shorts$, ^clip\d+$"):
            s = _build_settings("dev")
        self.assertEqual(s.opensearch.filename_noise_patterns, (r"^shorts$", r"^clip\d+$"))

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
    """search_service 검색 튜닝 필드 계약 봉인 — 필드명·기본값 정확 일치.

    PR4b: OS 경로가 ``SearchTuning.from_settings(cfg)`` → ``cfg.search.<field>`` 직접 중첩 접근으로 융합·
    게이트·컷 필드를 읽으므로, 하위 dataclass 필드명·기본값이 어긋나면 동작이 갈라진다(회귀). 이 계약을
    직접 봉인한다. 037: search.backend 는 settings 단일 경로 결정 필드로, 기본값이 'pg'→'opensearch' 로
    전환됐다(아래 단언). 027: 서버 파이프라인 메타 필드는 클라이언트 융합 전환으로 제거됐다 —
    제거 봉인은 아래 '제거된 필드 부재' 단언이 담당한다.
    """

    # 027·069 US-C: 제거된 설정 필드명(부재를 봉인). 리터럴을 문자열 결합으로 쪼개 한 곳에만 두어
    # 잔존 참조(grep)와 구분한다 — 삭제 심볼 grep 0(SC-C) 을 이 봉인이 깨지 않게 한다.
    _REMOVED_FIELDS = (
        "opensearch_" + "search_pipeline",
        "search_os_" + "probe_k",
        "search_os_" + "min_scores",
        # 069 US-C(037 잔재 철거): SEARCH_MIN_SCORE·chunk_agg 죽은 배선 삭제.
        "search_min" + "_scores",
        "chunk" + "_agg",
        "chunk" + "_agg_k",
        "chunk" + "_agg_mix_w",
        # 069 US-C(FR-C6): 037 로 소비자 소멸한 라벨 검색 top-k 2종(labels_score_min·*_meta_top_k 는 유지).
        "image_labels" + "_search_top_k",
        "video_keyframe_labels" + "_search_top_k",
    )

    def test_field_names_and_defaults_match_g3_getattr(self) -> None:
        with _env():
            settings = _build_settings("dev")
        # 필드명 일치(PR4b: 검색 튜닝 필드는 settings.search 하위로 중첩 — SearchTuning.from_settings 소비 계약)
        self.assertTrue(hasattr(settings.search, "backend"))
        self.assertTrue(hasattr(settings.search, "fusion_weights"))
        self.assertTrue(hasattr(settings.search, "os_result_floor"))
        # 027: 제거된 필드들은 더 이상 존재하지 않는다(서버 융합 파이프라인·정규화 스케일 임계 0)
        for removed in self._REMOVED_FIELDS:
            self.assertFalse(hasattr(settings, removed), f"제거된 필드가 잔존: {removed}")
        # 037: 기본 search_backend 가 'pg'→'opensearch' 로 전환됐다(PG 검색 경로 제거).
        self.assertEqual(settings.search.backend, "opensearch")
        self.assertEqual(settings.search.fusion_weights, search_constants.OS_FUSION_WEIGHTS_DEFAULT)
        self.assertEqual(settings.search.os_result_floor, search_constants.OS_RESULT_FLOOR_DEFAULT)


class TestRelationAutoApproveEmbMin(unittest.TestCase):
    """033 T002(FR-002): 자동승인 emb_score 하한 ``relation_auto_approve_emb_min`` —
    기본 0.0(무력=현 동작)·env ``RELATION_AUTO_APPROVE_EMB_MIN`` 오버라이드.

    기존 ``relation_auto_approve_min``(_env_float_default) 패턴을 그대로 미러. 기본 0.0 이라
    이 값이 0 이면 AND 게이트의 emb 변이가 무력화돼 LLM conf 단독 결정(현행 status 보존)."""

    def test_default_zero_when_unset(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertEqual(s.relations.auto_approve_emb_min, 0.0)

    def test_env_override(self) -> None:
        with _env(RELATION_AUTO_APPROVE_EMB_MIN="0.5"):
            s = _build_settings("dev")
        self.assertEqual(s.relations.auto_approve_emb_min, 0.5)


class TestSearchBackendWiring(unittest.TestCase):
    """T008 스모크(037 갱신): ``SEARCH_BACKEND`` 설정과 ``search_hybrid`` 경로 — OS 단일 백엔드.

    진입점(run_search·portal_api)은 ``backend`` 인자 없이 ``search_hybrid`` 를
    호출하며, 037 OpenSearch 전용 정리로 검색 경로는 OS 단일이다 — 즉 진입점 **호출부 코드 변경이
    불필요**함을 봉인한다(plan §B). settings 전역을 오염시키지 않도록 ``get_current_settings`` 를
    모킹해 빌드된 설정을 주입한다.
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

        # backend 인자 미전달(진입점 호출부와 동일) → settings.search.backend 가 경로를 결정한다.
        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "FAKE",
            )
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t", "similarity": 0.9}])
        self.assertEqual(os_cap["client"], "FAKE")

    def test_default_setting_routes_search_hybrid_to_os(self) -> None:
        # 037: SEARCH_BACKEND 미설정(기본 'opensearch') → backend 인자 없이도 OS seam 으로 라우팅된다.
        import src.search.search_service as svc

        with _env():  # SEARCH_BACKEND 미설정 → 기본 'opensearch'
            cfg = _build_settings("dev")
        self.assertEqual(cfg.search.backend, "opensearch")

        os_calls: list[object] = []

        def fake_os(
            client: object, query: str, **kw: object
        ) -> tuple[dict[str, list[dict[str, object]]], dict]:
            os_calls.append(1)
            return {"text": [{"id": "os_t"}]}, {}

        with mock.patch.object(svc, "get_current_settings", return_value=cfg):
            out = svc.search_hybrid(
                "질의",
                modalities=["text"],
                _os_search_fn=fake_os,
                _os_client_fn=lambda: "FAKE",
            )
        self.assertEqual(os_calls, [1])  # 기본값으로도 OS seam 사용(037 단일 경로)
        self.assertEqual(out["results"]["text_documents"], [{"id": "os_t"}])


class TestOpenSearchSyncGuard(unittest.TestCase):
    """038 T001/T002: OS read ⇒ OS write 정합 가드 + 증분 색인 기본 활성화.

    037 로 ``search_backend`` 가 OpenSearch 단일이 된 뒤, 적재 시 OS 증분 색인
    (``opensearch_sync_enabled``)이 꺼져 있으면 신규·변경 자산이 검색에서 조용히 누락된다(PG 폴백 없음).
    이 반쪽 마이그레이션 조합(``backend=opensearch ∧ ¬sync``)을 ``_build_settings`` 시점에 즉시
    ``ValueError`` 로 차단한다(``_resolve_search_backend`` 동형 fail-fast). 기본값은 ``True`` —
    미설정 환경도 적재=색인이 정합이다(가드와 한 쌍: 가드만 두고 기본 False 면 기본 설정 build 가 전부
    깨진다). ``_BACKEND_KEYS`` 가 ``OPENSEARCH_SYNC_ENABLED`` 를 비우므로 미설정=기본값 경로로 검증된다.
    """

    def test_default_sync_enabled_is_true(self) -> None:
        # SC-001: 미설정 → 기본 True(037 후 적재=색인). _BACKEND_KEYS 가 비워 실행 환경과 격리.
        with _env():
            settings = _build_settings("dev")
        self.assertIs(settings.opensearch.sync_enabled, True)

    def test_opensearch_backend_with_sync_off_raises(self) -> None:
        # SC-002: backend=opensearch(기본) ∧ sync=false → 빌드 시 즉시 ValueError(반쪽 마이그레이션 차단).
        with _env(OPENSEARCH_SYNC_ENABLED="false"):
            with self.assertRaises(ValueError) as cm:
                _build_settings("dev")
        msg = str(cm.exception)
        self.assertIn("OPENSEARCH_SYNC_ENABLED", msg)
        self.assertIn("SEARCH_BACKEND", msg)

    def test_consistent_on_builds_ok(self) -> None:
        # SC-002: 둘 다 일관(opensearch + true)이면 정상 빌드.
        with _env(OPENSEARCH_SYNC_ENABLED="true"):
            settings = _build_settings("dev")
        self.assertIs(settings.opensearch.sync_enabled, True)
        self.assertEqual(settings.search.backend, "opensearch")


class TestGenericTermSeedExtra(unittest.TestCase):
    """045 v2a — ``SEARCH_GENERIC_TERM_SEED_EXTRA`` env merge."""

    def test_default_is_core_seed_only(self) -> None:
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(
            settings.search.generic_term_seed,
            search_constants.GENERIC_SINGLE_TERM_SEED,
        )

    def test_extra_env_merges_and_dedups(self) -> None:
        with _env(SEARCH_GENERIC_TERM_SEED_EXTRA="foo,bar,TEST"):
            settings = _build_settings("dev")
        seed = settings.search.generic_term_seed
        self.assertIn("foo", seed)
        self.assertIn("bar", seed)
        self.assertEqual(len(seed), len(search_constants.GENERIC_SINGLE_TERM_SEED) + 2)

    def test_extra_seed_restricted_policy(self) -> None:
        from src.search import query_plan
        from src.search.query_plan import build_search_policy

        with _env(SEARCH_GENERIC_TERM_SEED_EXTRA="foo,bar"):
            settings = _build_settings("dev")
        with mock.patch.object(
            query_plan,
            "resolve_generic_term_seed",
            return_value=settings.search.generic_term_seed,
        ):
            p = build_search_policy("foo")
        self.assertTrue(p.generic_single_term)
        self.assertEqual(p.lexical_rescue, "restricted")

    def test_blank_extra_entry_raises(self) -> None:
        with _env(SEARCH_GENERIC_TERM_SEED_EXTRA="foo,,bar"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestVideoKeyframeDedupSettings(unittest.TestCase):
    """048 G0(FR-501): 영상 키프레임 near-dup 제거 7필드 단일 출처.

    기존 ``_env_*_default`` 선택 필드 패턴(021/023/038 동형) — 미설정 시 spec 기본설정표 값.
      - ``video_keyframe_dedup_enabled``     : 기본 True(사용자 결정 2026-06-29 · master switch).
      - ``video_keyframe_dedup_hash_max``    : 기본 7(64-bit Hamming 후보 상한).
      - ``video_keyframe_dedup_ssim_min``    : 기본 0.94(중복 확정 SSIM 하한).
      - ``video_keyframe_dedup_ssim_gray_lo``: 기본 0.90(히스토그램 보조 구간 하한).
      - ``video_keyframe_dedup_hist_min``    : 기본 0.97(HSV correlation 하한·보조).
      - ``video_keyframe_dedup_compare_mode``: 기본 'recent'('last'|'recent'|'global').
      - ``video_keyframe_dedup_recent_window``: 기본 4('recent' 모드 N).

    G2 까지 프로덕션 동작 무변경 — enabled=true 기본이라도 video_skill 배선(G3) 전이므로 검색·추출에
    아직 영향이 없다. 순수 토글·수치라 별도 범위검증 헬퍼 없이 ``_env_*_default`` 가 형식 오류만 fail-fast.
    """

    def test_defaults_when_unset(self) -> None:
        with _env():
            s = _build_settings("dev")
        self.assertIs(s.video.dedup_enabled, True)
        self.assertEqual(s.video.dedup_hash_max, 7)
        self.assertEqual(s.video.dedup_ssim_min, 0.94)
        self.assertEqual(s.video.dedup_ssim_gray_lo, 0.90)
        self.assertEqual(s.video.dedup_hist_min, 0.97)
        self.assertEqual(s.video.dedup_compare_mode, "recent")
        self.assertEqual(s.video.dedup_recent_window, 4)

    def test_enabled_env_override_false(self) -> None:
        with _env(VIDEO_KEYFRAME_DEDUP_ENABLED="false"):
            s = _build_settings("dev")
        self.assertIs(s.video.dedup_enabled, False)

    def test_numeric_env_overrides(self) -> None:
        with _env(
            VIDEO_KEYFRAME_DEDUP_HASH_MAX="5",
            VIDEO_KEYFRAME_DEDUP_SSIM_MIN="0.9",
            VIDEO_KEYFRAME_DEDUP_SSIM_GRAY_LO="0.85",
            VIDEO_KEYFRAME_DEDUP_HIST_MIN="0.95",
            VIDEO_KEYFRAME_DEDUP_RECENT_WINDOW="2",
        ):
            s = _build_settings("dev")
        self.assertEqual(s.video.dedup_hash_max, 5)
        self.assertEqual(s.video.dedup_ssim_min, 0.9)
        self.assertEqual(s.video.dedup_ssim_gray_lo, 0.85)
        self.assertEqual(s.video.dedup_hist_min, 0.95)
        self.assertEqual(s.video.dedup_recent_window, 2)

    def test_compare_mode_env_override(self) -> None:
        with _env(VIDEO_KEYFRAME_DEDUP_COMPARE_MODE="global"):
            s = _build_settings("dev")
        self.assertEqual(s.video.dedup_compare_mode, "global")

    def test_invalid_bool_fail_fast(self) -> None:
        # 불리언 형식 오류는 _env_bool_default 가 즉시 ValueError(잘못된 토글로 추출하지 않게).
        with _env(VIDEO_KEYFRAME_DEDUP_ENABLED="maybe"):
            with self.assertRaises(ValueError):
                _build_settings("dev")

    def test_invalid_int_fail_fast(self) -> None:
        # 정수 형식 오류는 _env_int_default 가 즉시 ValueError.
        with _env(VIDEO_KEYFRAME_DEDUP_HASH_MAX="abc"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestVlmSummaryPromptV2Settings(unittest.TestCase):
    """049 G0(FR-601): VLM 요약 프롬프트 v2 토글 2종 단일 출처.

    기존 ``_env_bool_default`` 선택 필드 패턴(029/038/048 동형) — 미설정 시 기본 False.
      - ``vlm_summary_prompt_v2``: 기본 False(v1 바이트 동일·회귀 안전판·FR-102). True 면 캡션·reduce
        v2 프롬프트 + 키워드 후처리(objects 승격) 활성.
      - ``vlm_summary_ab_judge`` : 기본 False(A/B 하니스 LLM-judge 옵션).

    순수 토글이라 별도 범위검증 헬퍼 없이 ``_env_bool_default`` 가 불리언 형식 오류만 fail-fast.
    """

    def test_defaults_when_unset(self) -> None:
        # 미설정 → 둘 다 기본 False(v1 바이트 동일·회귀 안전판). _BACKEND_KEYS 가 비워 실행 환경과 격리.
        with _env():
            s = _build_settings("dev")
        self.assertIs(s.vlm.summary_prompt_v2, False)
        self.assertIs(s.vlm.summary_ab_judge, False)

    def test_prompt_v2_env_override_true(self) -> None:
        with _env(VLM_SUMMARY_PROMPT_V2="true"):
            s = _build_settings("dev")
        self.assertIs(s.vlm.summary_prompt_v2, True)

    def test_ab_judge_env_override_true(self) -> None:
        with _env(VLM_SUMMARY_AB_JUDGE="1"):
            s = _build_settings("dev")
        self.assertIs(s.vlm.summary_ab_judge, True)

    def test_prompt_v2_env_override_false_explicit(self) -> None:
        with _env(VLM_SUMMARY_PROMPT_V2="off"):
            s = _build_settings("dev")
        self.assertIs(s.vlm.summary_prompt_v2, False)

    def test_invalid_bool_fail_fast(self) -> None:
        # 불리언 형식 오류는 _env_bool_default 가 즉시 ValueError(잘못된 토글로 추출하지 않게).
        with _env(VLM_SUMMARY_PROMPT_V2="maybe"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestTopicCanonicalizeEnabledSettings(unittest.TestCase):
    """058 G4(FR-401): 관계 topic 정본화 배선 토글 ``topic_canonicalize_enabled``.

    기존 ``_env_bool_default`` 선택 필드 패턴(029/049 동형) — 미설정 시 기본 **False**(동작 불변).
    빈 레지스트리에서 canonicalize 를 켜면 raw topic 이 전부 자동등록(부작용)돼 시드 전 동작이 깨지므로
    기본 off 로 두고, 시드(G5) 후 명시적으로 켠다. 순수 토글이라 불리언 형식 오류만 fail-fast.
    """

    def test_default_off_when_unset(self) -> None:
        # 미설정 → 기본 False(동작 불변·시드 전 동치). _BACKEND_KEYS 가 비워 실행 환경과 격리.
        with _env():
            s = _build_settings("dev")
        self.assertIs(s.topic.canonicalize_enabled, False)

    def test_env_override_true(self) -> None:
        with _env(TOPIC_CANONICALIZE_ENABLED="true"):
            s = _build_settings("dev")
        self.assertIs(s.topic.canonicalize_enabled, True)

    def test_env_override_false_explicit(self) -> None:
        with _env(TOPIC_CANONICALIZE_ENABLED="off"):
            s = _build_settings("dev")
        self.assertIs(s.topic.canonicalize_enabled, False)

    def test_invalid_bool_fail_fast(self) -> None:
        with _env(TOPIC_CANONICALIZE_ENABLED="maybe"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestEmbedEnableClipSettings(unittest.TestCase):
    """063(FR-101): image/video CLIP 임베딩 토글 ``embed_enable_clip``.

    ``_env_bool_default`` 선택 필드 — 미설정 시 기본 **True**(회귀 0·기존 동작 불변). 신규 셋업서만
    false 로 opt-out. 순수 토글이라 불리언 형식 오류만 fail-fast.
    """

    def test_default_on_when_unset(self) -> None:
        # 미설정 → 기본 True(회귀 0). _BACKEND_KEYS 가 비워 실행 환경과 격리.
        with _env():
            s = _build_settings("dev")
        self.assertIs(s.embed.enable_clip, True)

    def test_env_override_false(self) -> None:
        with _env(EMBED_ENABLE_CLIP="false"):
            s = _build_settings("dev")
        self.assertIs(s.embed.enable_clip, False)

    def test_env_override_true_explicit(self) -> None:
        with _env(EMBED_ENABLE_CLIP="on"):
            s = _build_settings("dev")
        self.assertIs(s.embed.enable_clip, True)

    def test_invalid_bool_fail_fast(self) -> None:
        with _env(EMBED_ENABLE_CLIP="maybe"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


class TestFieldSpecsSSOT(unittest.TestCase):
    """069 US-E FR-E4 — 필드 명세 단일 출처(``_FIELD_SPECS``·그룹 포함).

    build 조립(그룹별 하위 dataclass)·테스트 격리키(``_BACKEND_KEYS``)·커버리지가 모두 이 한 테이블에서
    파생돼야 한다. 새 env 필드 추가 시 테이블 한 행만 늘리면 build·격리·검증이 자동 반영된다(SC-E: 필드
    추가 시 테스트 수정 0). 이 계약이 깨지면(테이블 누락/드리프트) 아래 테스트가 즉시 실패한다."""

    def test_specs_cover_all_dataclass_fields(self) -> None:
        # 각 하위 dataclass 필드 ↔ 해당 그룹 spec 이 정확히 1:1, 상위 공통(group="")도 1:1(PR4b 중첩).
        from dataclasses import fields

        from src.config.settings import _FIELD_SPECS, _GROUP_CLASSES, PipelineSettings

        by_group: dict[str, list[str]] = {}
        for s in _FIELD_SPECS:
            by_group.setdefault(s.group, []).append(s.attr)
        for g, attrs in by_group.items():
            self.assertEqual(len(attrs), len(set(attrs)), f"그룹 {g!r} 중복 attr")
        # 하위 dataclass 그룹
        for name, cls in _GROUP_CLASSES.items():
            self.assertEqual(
                set(by_group.get(name, [])),
                {f.name for f in fields(cls)},
                f"그룹 {name!r} spec ↔ dataclass 불일치",
            )
        # 상위 공통 = PipelineSettings 필드 - profile - 그룹 필드
        common = {f.name for f in fields(PipelineSettings)} - {"profile"} - set(_GROUP_CLASSES)
        self.assertEqual(set(by_group.get("", [])), common, "상위 공통 spec ↔ dataclass 불일치")

    def test_backend_keys_derived_from_specs(self) -> None:
        # 테스트 격리키(선택 env)는 수동 나열이 아니라 테이블의 required=False 행에서 파생된다.
        from src.config.settings import _FIELD_SPECS

        optional_env = {s.env for s in _FIELD_SPECS if not s.required}
        self.assertEqual(set(_BACKEND_KEYS), optional_env)

    def test_required_flag_matches_reader(self) -> None:
        # required 플래그 ⇔ read 함수 계열 봉인: 필수 필드는 _require_env* 로, 선택 필드는 그 외
        # (_opt_*/resolver/특이 read)로 읽어야 한다. 종전엔 _require_env* 호출 자체가 곧 필수 여부라
        # 메타-동작이 붙어 있었으나, 테이블화로 분리되며 "required=True 인데 _opt_*" 같은 오설정
        # 여지가 생긴다 — 이 상관을 명시적으로 잠가 미래 footgun 을 차단한다(리뷰 반영).
        from src.config.settings import _FIELD_SPECS

        for s in _FIELD_SPECS:
            uses_require_reader = s.read.__name__.startswith("_require_")
            self.assertEqual(
                s.required,
                uses_require_reader,
                f"{s.attr}: required={s.required} 인데 read={s.read.__name__} "
                "(필수 필드는 _require_env* 로 읽어야 한다)",
            )


if __name__ == "__main__":
    unittest.main()
