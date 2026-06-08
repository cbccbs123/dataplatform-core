"""활성 임베딩 채널 프로파일 헬퍼 — 순수 단위 테스트(DB·네트워크 0).

018 은 운영 텍스트 임베딩 채널(=모델)을 ``EMBED_ACTIVE_CHANNEL`` 설정으로 토글한다.
적재·검색·관계가 흩어진 'st' 하드코딩 대신 **활성 채널 단일 출처**를 참조하도록,
``settings.active_embed_channel`` 필드와 모듈 헬퍼 두 개를 제공한다:

- ``active_embed_channel(settings)`` → 활성 채널 문자열(기본 'st')
- ``active_embed_model(settings)`` → 활성 채널의 임베딩 모델(017 ``model_for_channel`` 재사용)

순수 단위 전략은 017 ``test_embed_model_map`` 과 동형 — ``_build_settings`` 로
``PipelineSettings`` 를 직접 만들어(필수 env 임시 주입) 헬퍼에 주입한다. ``init_settings``/DB 없이
기본값·채널·모델·미지원 채널(ValueError)을 검증한다. **기본 active='st' → 회귀 0** 임을 함께 확인한다.
"""

from __future__ import annotations

import contextlib
import os
import unittest

from src.config.settings import (
    _build_settings,
    active_embed_channel,
    active_embed_model,
)

# _build_settings 가 _require_env* 로 읽는 필수 env 최소 집합(값은 형식만 맞으면 됨).
# TEXT_EMBED_MODEL 은 모델 매핑 검증을 위해 실제 KoSimCSE 값으로 고정한다.
_KOSIMCSE = "BM-K/KoSimCSE-roberta-multitask"
_BGE = "BAAI/bge-m3"
_REQUIRED_ENV = {
    "META_MODEL": "gemma",
    "OPENAI_BASE_URL": "http://localhost:1234/v1",
    "OPENAI_API_KEY": "sk-test",
    "SUMMARY_MAX_CHARS": "500",
    "TOP_K_KEYWORDS": "10",
    "CHUNK_SIZE": "1000",
    "OVERLAP_SIZE": "100",
    "ENCODING": "utf-8",
    "TEXT_EMBED_MODEL": _KOSIMCSE,
    "TEXT_EMBED_CHUNK_SIZE": "512",
    "TEXT_EMBED_NORMALIZE": "true",
}

_ACTIVE_KEY = "EMBED_ACTIVE_CHANNEL"


@contextlib.contextmanager
def _env(active: str | None = None):
    """필수 env 를 임시로 채우고 ``EMBED_ACTIVE_CHANNEL`` 을 ``active`` 로 설정(None=미설정).

    본 테스트가 건드린 키만 정확히 원복한다(필수 env 가 원래 있었으면 보존).
    """
    touched = list(_REQUIRED_ENV) + [_ACTIVE_KEY]
    saved = {k: os.environ.get(k) for k in touched}
    try:
        os.environ.update(_REQUIRED_ENV)
        os.environ.pop(_ACTIVE_KEY, None)
        if active is not None:
            os.environ[_ACTIVE_KEY] = active
        yield
    finally:
        for k in touched:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


class TestActiveEmbedChannelField(unittest.TestCase):
    def test_default_is_st_when_unset(self) -> None:
        # 선택 필드: 미설정 시 기본 'st' (필수 env 가 아니라 누락해도 ValueError 안 남). → 회귀 0.
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.active_embed_channel, "st")

    def test_env_override(self) -> None:
        # 설정 시 그 값으로 해석(운영 토글: st_bge 로 전환).
        with _env(active="st_bge"):
            settings = _build_settings("dev")
        self.assertEqual(settings.active_embed_channel, "st_bge")


class TestActiveEmbedChannelHelper(unittest.TestCase):
    def test_returns_settings_active_channel(self) -> None:
        # 헬퍼는 주입 settings 의 활성 채널을 그대로 반환(단일 출처).
        with _env(active="st_bge"):
            settings = _build_settings("dev")
        self.assertEqual(active_embed_channel(settings), "st_bge")

    def test_default_st(self) -> None:
        # 기본(미설정) → 'st' (회귀 가드).
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(active_embed_channel(settings), "st")


class TestActiveEmbedModelHelper(unittest.TestCase):
    def test_st_active_maps_to_kosimcse(self) -> None:
        # 기본 active='st' → model_for_channel('st') = KoSimCSE. 회귀 0(기존 적재·검색 모델 불변).
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(active_embed_model(settings), _KOSIMCSE)

    def test_st_bge_active_maps_to_bge(self) -> None:
        # active='st_bge' → model_for_channel('st_bge') = BGE-M3.
        with _env(active="st_bge"):
            settings = _build_settings("dev")
        self.assertEqual(active_embed_model(settings), _BGE)

    def test_unsupported_active_raises_value_error(self) -> None:
        # 미지원 채널은 model_for_channel 이 즉시 ValueError(잘못된 모델 사용 차단).
        with _env(active="zzz"):
            settings = _build_settings("dev")
        with self.assertRaises(ValueError):
            active_embed_model(settings)


class TestFallsBackToCurrentSettings(unittest.TestCase):
    def test_none_uses_current_settings(self) -> None:
        # settings 인자를 주지 않으면 get_current_settings() 의 활성 설정을 쓴다(운영 호출 경로).
        import src.config.settings as settings_mod

        saved_global = settings_mod._SETTINGS
        try:
            with _env(active="st_bge"):
                settings_mod.init_settings("dev")
                self.assertEqual(active_embed_channel(), "st_bge")
                self.assertEqual(active_embed_model(), _BGE)
        finally:
            settings_mod._SETTINGS = saved_global


if __name__ == "__main__":
    unittest.main()
