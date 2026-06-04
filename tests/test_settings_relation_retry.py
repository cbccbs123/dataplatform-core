"""관계 재시도 상한 운영 파라미터 ``relation_retry_max_attempts`` — 순수 단위 테스트.

``PipelineSettings`` 에 ``RELATION_RETRY_MAX_ATTEMPTS``(기본 3) 선택 환경변수를 추가했다.
기존 ``relation_top_k``/``relation_min_sim`` 과 같은 ``_env_int_default`` 패턴이므로, 미설정 시
기본값 3, 설정 시 그 값으로 해석되는지 ``_build_settings`` 를 직접 호출해 검증한다.

``_build_settings`` 는 17개 필수 env 를 요구하므로, 본 테스트는 그 최소 env 를 임시로 채운 뒤
``RELATION_RETRY_MAX_ATTEMPTS`` 만 토글한다(다른 테스트 환경을 오염시키지 않도록 복원).
"""

from __future__ import annotations

import contextlib
import os
import unittest

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

_RETRY_KEY = "RELATION_RETRY_MAX_ATTEMPTS"


@contextlib.contextmanager
def _env(retry: str | None = None):
    """필수 env 를 임시로 채우고 ``RELATION_RETRY_MAX_ATTEMPTS`` 를 ``retry`` 로 설정(None=미설정).

    실행 후 본 테스트가 건드린 키만 정확히 원복한다(필수 env 가 원래 있었으면 보존).
    """
    touched = list(_REQUIRED_ENV) + [_RETRY_KEY]
    saved = {k: os.environ.get(k) for k in touched}
    try:
        os.environ.update(_REQUIRED_ENV)
        os.environ.pop(_RETRY_KEY, None)
        if retry is not None:
            os.environ[_RETRY_KEY] = retry
        yield
    finally:
        for k in touched:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


class TestRelationRetryMaxAttempts(unittest.TestCase):
    def test_default_is_three_when_unset(self) -> None:
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.relation_retry_max_attempts, 3)

    def test_env_override(self) -> None:
        with _env(retry="5"):
            settings = _build_settings("dev")
        self.assertEqual(settings.relation_retry_max_attempts, 5)

    def test_invalid_int_env_raises(self) -> None:
        # 잘못된 형식의 정수 env 는 _env_int_default 가 ValueError — 프로세스 시작 시 빠르게 실패.
        with _env(retry="not_a_number"):
            with self.assertRaises(ValueError):
                _build_settings("dev")


if __name__ == "__main__":
    unittest.main()
