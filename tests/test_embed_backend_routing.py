"""062 G3 — 채널 백엔드 라우팅 단위 테스트 [FR-103·104·SC-04·07].

``embed_texts_for``(적재·질의 공유 라우터)가 채널의 backend 로 로컬↔API 를 고르는지 mock 으로 검증한다.
기본 로컬 채널(st/st_bge)은 기존 ``embed_texts`` 를 그대로 호출(동작 불변·회귀 0), ``st_api`` 만 API.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import unittest
from unittest.mock import patch

from src.config.settings import _build_settings, _validate_settings_consistency
from src.embedders import text_embedder
from src.search import query_embed

_REQUIRED_ENV = {
    "META_MODEL": "gemma", "OPENAI_BASE_URL": "http://localhost:1234/v1",
    "OPENAI_API_KEY": "sk-test", "SUMMARY_MAX_CHARS": "500", "TOP_K_KEYWORDS": "10",
    "CHUNK_SIZE": "1000", "OVERLAP_SIZE": "100", "ENCODING": "utf-8",
    "TEXT_EMBED_MODEL": "BM-K/KoSimCSE-roberta-multitask",
    "TEXT_EMBED_CHUNK_SIZE": "512", "TEXT_EMBED_NORMALIZE": "true",
}


@contextlib.contextmanager
def _env(**extra: str):
    touched = list(_REQUIRED_ENV) + list(extra)
    saved = {k: os.environ.get(k) for k in touched}
    try:
        os.environ.update(_REQUIRED_ENV)
        os.environ.update(extra)
        yield
    finally:
        for k in touched:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


def _settings(**extra: str):
    with _env(**extra):
        return _build_settings("dev")


class TestEmbedTextsForRouting(unittest.TestCase):
    def test_routes_to_api_for_st_api(self) -> None:
        cfg = _settings(EMBED_API_BASE_URL="http://x/v1", EMBED_API_MODEL="BAAI/bge-m3")
        with patch("src.embedders.text_embedder_api.embed_texts_api", return_value=[[1.0]]) as api, \
             patch.object(text_embedder, "embed_texts") as local:
            out = text_embedder.embed_texts_for(["q"], channel="st_api", settings=cfg)
        self.assertEqual(out, [[1.0]])
        local.assert_not_called()
        self.assertEqual(api.call_args.kwargs["base_url"], "http://x/v1")
        self.assertEqual(api.call_args.kwargs["model"], "BAAI/bge-m3")

    def test_routes_to_local_for_st_bge(self) -> None:
        cfg = _settings()
        with patch.object(text_embedder, "embed_texts", return_value=[[2.0]]) as local, \
             patch("src.embedders.text_embedder_api.embed_texts_api") as api:
            out = text_embedder.embed_texts_for(["q"], channel="st_bge", settings=cfg)
        self.assertEqual(out, [[2.0]])
        api.assert_not_called()
        self.assertEqual(local.call_args.kwargs["model_name"], "BAAI/bge-m3")

    def test_api_key_none_when_empty(self) -> None:
        cfg = _settings(EMBED_API_BASE_URL="http://x/v1")  # EMBED_API_KEY 미설정=""
        with patch("src.embedders.text_embedder_api.embed_texts_api", return_value=[[0.0]]) as api:
            text_embedder.embed_texts_for(["q"], channel="st_api", settings=cfg)
        self.assertIsNone(api.call_args.kwargs["api_key"])


class TestQueryEmbedRouting(unittest.TestCase):
    def test_query_routes_api_for_st_api(self) -> None:
        cfg = _settings(EMBED_API_BASE_URL="http://x/v1")
        with patch.object(query_embed, "get_current_settings", return_value=cfg), \
             patch.object(query_embed, "embed_texts_for", return_value=[[1.0, 0.0]]) as api, \
             patch.object(query_embed, "embed_texts") as local:
            query_embed.embed_query_for_media_search("요리", model_name="BAAI/bge-m3", channel="st_api")
        api.assert_called_once()
        local.assert_not_called()

    def test_query_routes_local_for_st(self) -> None:
        cfg = _settings()
        with patch.object(query_embed, "get_current_settings", return_value=cfg), \
             patch.object(query_embed, "embed_texts_for") as api, \
             patch.object(query_embed, "embed_texts", return_value=[[3.0]]) as local:
            query_embed.embed_query_for_media_search("요리", channel="st")
        api.assert_not_called()
        local.assert_called_once()


class TestApiChannelFailFast(unittest.TestCase):
    """062: active=st_api 인데 base_url 비면 기동 시점 fail-fast(038 관례·파이프라인 중단 방지)."""

    def test_st_api_without_base_url_raises(self) -> None:
        base = _settings(EMBED_API_BASE_URL="http://x/v1")
        bad = dataclasses.replace(base, active_embed_channel="st_api", embed_api_base_url="")
        with self.assertRaises(ValueError):
            _validate_settings_consistency(bad)

    def test_st_api_with_base_url_ok(self) -> None:
        base = _settings(EMBED_API_BASE_URL="http://x/v1")
        ok = dataclasses.replace(
            base, active_embed_channel="st_api", embed_api_base_url="http://x/v1"
        )
        _validate_settings_consistency(ok)  # 예외 없음

    def test_local_channel_without_base_url_ok(self) -> None:
        # 기본 로컬 채널은 base_url 없어도 정상(회귀 0).
        base = _settings()
        local = dataclasses.replace(base, active_embed_channel="st", embed_api_base_url="")
        _validate_settings_consistency(local)


if __name__ == "__main__":
    unittest.main()
