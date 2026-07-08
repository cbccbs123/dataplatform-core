"""채널→모델 매핑 헬퍼 + BGE 임베딩 모델 설정 필드 — 순수 단위 테스트(DB·네트워크 0).

017 A/B 는 같은 자산을 KoSimCSE(``channel='st'``)와 BGE-M3(``channel='st_bge'``) 두 채널로
임베딩한다. 질의 시 채널에 맞는 임베딩 모델을 골라야 질의-문서 모델이 일치(FR-004)하므로,
채널→모델 매핑을 설정에서 단일 출처로 제공하고 **미지원 채널은 즉시 ValueError** 로 차단한다.

순수 단위 전략: ``_build_settings`` 로 ``PipelineSettings`` 를 직접 만들어(필수 env 임시 주입)
헬퍼에 주입한다. ``init_settings``/DB 없이 매핑·기본값·미지원 채널을 검증한다.
``text_embedding_model_bge`` 는 ``_require_env`` 가 아닌 **선택 필드**(미설정 시 ``BAAI/bge-m3``)임도 함께 확인한다.
"""

from __future__ import annotations

import contextlib
import os
import unittest

from src.config.settings import _build_settings, backend_for_channel, model_for_channel

# _build_settings 가 _require_env* 로 읽는 필수 env 최소 집합(값은 형식만 맞으면 됨).
# TEXT_EMBED_MODEL 은 매핑 검증을 위해 실제 KoSimCSE 값으로 고정한다.
_KOSIMCSE = "BM-K/KoSimCSE-roberta-multitask"
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

_BGE_KEY = "TEXT_EMBED_MODEL_BGE"


@contextlib.contextmanager
def _env(bge: str | None = None):
    """필수 env 를 임시로 채우고 ``TEXT_EMBED_MODEL_BGE`` 를 ``bge`` 로 설정(None=미설정).

    본 테스트가 건드린 키만 정확히 원복한다(필수 env 가 원래 있었으면 보존).
    """
    touched = list(_REQUIRED_ENV) + [_BGE_KEY]
    saved = {k: os.environ.get(k) for k in touched}
    try:
        os.environ.update(_REQUIRED_ENV)
        os.environ.pop(_BGE_KEY, None)
        if bge is not None:
            os.environ[_BGE_KEY] = bge
        yield
    finally:
        for k in touched:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


class TestTextEmbeddingModelBgeField(unittest.TestCase):
    def test_default_is_bge_m3_when_unset(self) -> None:
        # 선택 필드: 미설정 시 기본 BAAI/bge-m3 (필수 env 가 아니라 누락해도 ValueError 안 남).
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.text_embedding_model_bge, "BAAI/bge-m3")

    def test_env_override(self) -> None:
        # 설정 시 그 값으로 해석(향후 다른 BGE 변형 모델 교체 여지).
        with _env(bge="org/bge-custom"):
            settings = _build_settings("dev")
        self.assertEqual(settings.text_embedding_model_bge, "org/bge-custom")

    def test_kosimcse_field_unchanged(self) -> None:
        # 회귀 가드: 기존 text_embedding_model(KoSimCSE) 동작은 무변경.
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(settings.text_embedding_model, _KOSIMCSE)


class TestModelForChannel(unittest.TestCase):
    def test_st_channel_maps_to_kosimcse(self) -> None:
        # 'st' 채널 질의는 기존 KoSimCSE 모델로 임베딩해야 질의-문서 모델 일치(FR-004).
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(model_for_channel("st", settings), _KOSIMCSE)

    def test_st_bge_channel_maps_to_bge(self) -> None:
        # 'st_bge' 채널 질의는 BGE-M3 로 임베딩해야 한다.
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(model_for_channel("st_bge", settings), "BAAI/bge-m3")

    def test_unsupported_channel_raises_value_error(self) -> None:
        # 미지원 채널은 조용히 잘못된 모델로 검색하지 않고 즉시 ValueError(명확 차단).
        with _env():
            settings = _build_settings("dev")
        for bad in ("zzz", "clip", ""):
            with self.subTest(channel=bad):
                with self.assertRaises(ValueError):
                    model_for_channel(bad, settings)

    def test_falls_back_to_current_settings_when_none(self) -> None:
        # settings 인자를 주지 않으면 get_current_settings() 의 활성 설정을 쓴다(운영 호출 경로).
        import src.config.settings as settings_mod

        saved_global = settings_mod._SETTINGS
        try:
            with _env(bge="org/bge-active"):
                settings_mod.init_settings("dev")
                self.assertEqual(model_for_channel("st_bge"), "org/bge-active")
                self.assertEqual(model_for_channel("st"), _KOSIMCSE)
        finally:
            settings_mod._SETTINGS = saved_global


class TestApiEmbedChannel(unittest.TestCase):
    """062: st_api 채널 매핑 + backend_for_channel(로컬↔API 직교 축)."""

    def test_st_api_maps_to_embed_api_model_default(self) -> None:
        # 채널→모델: 'st_api' → embed_api_model(기본 'BAAI/bge-m3'·vLLM 서버 모델 id·GET /v1/models 실측).
        with _env():
            settings = _build_settings("dev")
        self.assertEqual(model_for_channel("st_api", settings), "BAAI/bge-m3")

    def test_backend_for_channel(self) -> None:
        # 'st_api'만 API 백엔드, 나머지는 로컬(기본 st=로컬 → 동작 불변).
        self.assertEqual(backend_for_channel("st_api"), "api")
        self.assertEqual(backend_for_channel("st"), "local")
        self.assertEqual(backend_for_channel("st_bge"), "local")


if __name__ == "__main__":
    unittest.main()
