"""per-asset 청크 집계 설정(019) — 단일 출처 헬퍼 ``chunk_agg_config`` 순수 단위 테스트.

019 는 검색 시점 per-asset 청크 집계 방식(``max``|``topk_mean``|``mix``)을
``SEARCH_CHUNK_AGG``/``_K``/``_MIX_W`` 설정으로 토글한다(017/018 패턴). 검색 경로(ST 하이브리드·
시각 2단계)가 흩어진 ``MAX`` 하드코딩 대신 **집계 설정 단일 출처**를 참조하도록,
``settings`` 에 3종 필드를 더하고 모듈 헬퍼 ``chunk_agg_config(settings)`` 를 제공한다.

순수 단위 전략은 018 ``test_active_embed_profile`` 과 동형 — ``_build_settings`` 로
``PipelineSettings`` 를 직접 만들어(필수 env 임시 주입) 헬퍼에 주입한다. ``init_settings``/DB 없이
기본값(``max``,3,0.5)·env 오버라이드·미지원 값(ValueError)·미초기화 보수 폴백(``max``)·결정성을
검증한다. **기본 ``SEARCH_CHUNK_AGG=max`` → 회귀 0**(SC-001) 임을 함께 확인한다.
"""

from __future__ import annotations

import contextlib
import os
import unittest
from unittest import mock

from src.config.settings import _build_settings, chunk_agg_config

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
    "TEXT_EMBED_MODEL": "BM-K/KoSimCSE-roberta-multitask",
    "TEXT_EMBED_CHUNK_SIZE": "512",
    "TEXT_EMBED_NORMALIZE": "true",
}

_CHUNK_KEYS = ("SEARCH_CHUNK_AGG", "SEARCH_CHUNK_AGG_K", "SEARCH_CHUNK_AGG_MIX_W")


@contextlib.contextmanager
def _env(**overrides: str):
    """필수 env 를 임시로 채우고 SEARCH_CHUNK_AGG* 키는 비운 뒤 ``overrides`` 만 설정·복원한다.

    본 테스트가 건드린 키만 정확히 원복한다(필수 env 가 원래 있었으면 보존).
    """
    touched = list(_REQUIRED_ENV) + list(_CHUNK_KEYS)
    saved = {k: os.environ.get(k) for k in touched}
    try:
        os.environ.update(_REQUIRED_ENV)
        for k in _CHUNK_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: str(v) for k, v in overrides.items()})
        yield
    finally:
        for k in touched:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


class TestChunkAggConfigDefaults(unittest.TestCase):
    def test_default_is_max_3_half_when_unset(self) -> None:
        # 선택 필드: 미설정 시 기본 (max, 3, 0.5). 기본 max = 기존 MAX 집계와 동치 → 회귀 0(SC-001).
        with _env():
            settings = _build_settings("dev")
        cfg = chunk_agg_config(settings)
        self.assertEqual(cfg.agg, "max")
        self.assertEqual(cfg.k, 3)
        self.assertEqual(cfg.mix_w, 0.5)

    def test_k_and_mix_w_default_under_nondefault_agg(self) -> None:
        # agg 만 바꾸고 K/MIX_W 미지정 → 각 기본(3, 0.5)으로 폴백.
        with _env(SEARCH_CHUNK_AGG="topk_mean"):
            settings = _build_settings("dev")
        cfg = chunk_agg_config(settings)
        self.assertEqual(cfg.agg, "topk_mean")
        self.assertEqual(cfg.k, 3)
        self.assertEqual(cfg.mix_w, 0.5)


class TestChunkAggConfigEnvOverride(unittest.TestCase):
    def test_topk_mean_with_k(self) -> None:
        with _env(SEARCH_CHUNK_AGG="topk_mean", SEARCH_CHUNK_AGG_K="5"):
            settings = _build_settings("dev")
        cfg = chunk_agg_config(settings)
        self.assertEqual(cfg.agg, "topk_mean")
        self.assertEqual(cfg.k, 5)

    def test_mix_with_weight(self) -> None:
        with _env(SEARCH_CHUNK_AGG="mix", SEARCH_CHUNK_AGG_MIX_W="0.7"):
            settings = _build_settings("dev")
        cfg = chunk_agg_config(settings)
        self.assertEqual(cfg.agg, "mix")
        self.assertEqual(cfg.mix_w, 0.7)

    def test_max_is_supported_value(self) -> None:
        # 명시 max 도 유효(미지원 값 차단이 정상 값을 막지 않음).
        with _env(SEARCH_CHUNK_AGG="max"):
            settings = _build_settings("dev")
        self.assertEqual(chunk_agg_config(settings).agg, "max")


class TestChunkAggConfigUnsupported(unittest.TestCase):
    def test_unsupported_agg_raises_value_error(self) -> None:
        # 미지원 집계 방식은 헬퍼가 즉시 ValueError(잘못된 산식으로 검색하지 않도록 차단).
        with _env(SEARCH_CHUNK_AGG="zzz"):
            settings = _build_settings("dev")
        with self.assertRaises(ValueError):
            chunk_agg_config(settings)


class TestChunkAggConfigUninitializedFallback(unittest.TestCase):
    """settings 미초기화 시 보수적으로 ``max`` 기본 집계로 폴백(FR-006) — 회귀 0 가드."""

    def test_uninitialized_settings_falls_back_to_max(self) -> None:
        # get_current_settings 가 RuntimeError(미초기화)면 (max, 3, 0.5) 보수 폴백 — 기존 MAX 동작.
        with mock.patch(
            "src.config.settings.get_current_settings",
            side_effect=RuntimeError("settings 미초기화"),
        ):
            cfg = chunk_agg_config()
        self.assertEqual(cfg.agg, "max")
        self.assertEqual(cfg.k, 3)
        self.assertEqual(cfg.mix_w, 0.5)


class TestChunkAggConfigDeterminism(unittest.TestCase):
    def test_same_settings_returns_equal_config_twice(self) -> None:
        # 결정성(헌법 3조): 같은 설정 2회 호출 → 동일 반환.
        with _env(SEARCH_CHUNK_AGG="mix", SEARCH_CHUNK_AGG_K="4", SEARCH_CHUNK_AGG_MIX_W="0.3"):
            settings = _build_settings("dev")
        self.assertEqual(chunk_agg_config(settings), chunk_agg_config(settings))


if __name__ == "__main__":
    unittest.main()
