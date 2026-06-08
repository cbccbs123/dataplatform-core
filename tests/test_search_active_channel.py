"""018 G3 — 검색 활성 임베딩 채널 프로파일(search_service.search_hybrid) 단위 테스트.

018 은 운영 텍스트 임베딩 채널(=모델)을 ``EMBED_ACTIVE_CHANNEL`` 설정으로 토글한다.
검색은 채널·질의모델을 **미지정(None)** 으로 두면 활성 프로파일(적재·검색·관계 단일 출처)로
해소하고, 명시 전달(017 A/B 하니스)은 그대로 우선한다.

테스트 전략(docs/테스트_가이드.md §2)
  - **grouped seam 스파이**: 실제 검색(``search_media_all_grouped``)·DB·임베딩 모델 없이
    ``_grouped_fn`` 을 주입해 ``search_hybrid`` 가 channel·query_model_name 을 grouped 경로로
    **무엇을 전달하는지**만 검증한다.
  - **settings 모킹**: 활성 해소가 ``get_current_settings`` 를 거치므로 가짜 설정을 주입해
    순수 단위로 채널·모델을 검증한다(``init_settings``/DB 0).

핵심 가드
  - ① **미지정(None) → 활성 해소**: 기본 active='st' 시 channel='st'·KoSimCSE 전달(회귀 가드).
  - ② **active='st_bge' → st_bge·BGE 전달**(질의-문서 모델 일치, FR-004).
  - ③ **명시 우선(A/B 호환)**: text_channel/text_query_model 을 넘기면 활성과 무관하게 그 값.
  - ④ **settings 미초기화 폴백**: settings 없이도 기본 'st'·None(=media_search 가 KoSimCSE 해소)
    으로 동작 — 기존 검색 단위(006/017)가 settings 없이 그대로 도는 회귀 가드.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.search.search_service import search_hybrid

_KOSIMCSE = "BM-K/KoSimCSE-roberta-multitask"
_BGE = "BAAI/bge-m3"


def _fake_cfg(active: str) -> SimpleNamespace:
    """active_embed_channel/model_for_channel 이 읽는 필드만 가진 가짜 설정.

    실 .env·init_settings 없이 활성 채널·모델 매핑을 돌린다(순수 단위).
    """
    return SimpleNamespace(
        active_embed_channel=active,
        text_embedding_model=_KOSIMCSE,
        text_embedding_model_bge=_BGE,
    )


def _capturing_grouped():
    """grouped 호출 kwargs 를 잡아 두는 스파이(검색·DB 무관, 빈 버킷 반환)."""
    captured: dict[str, object] = {}

    def _g(query: str, **kw: object) -> dict[str, object]:
        captured.update(kw)
        return {"text_documents": [], "audio": [], "image": [], "video": [], "meta": {}}

    return _g, captured


class TestSearchActiveChannelDefault(unittest.TestCase):
    """미지정(None) → 활성 채널·모델로 해소(단일 출처)."""

    def test_none_resolves_active_st_to_kosimcse(self) -> None:
        # ① 기본 active='st' → grouped 에 channel='st' + KoSimCSE 전달(회귀 가드, SC-002).
        g, cap = _capturing_grouped()
        with mock.patch(
            "src.config.settings.get_current_settings", return_value=_fake_cfg("st")
        ):
            search_hybrid("질의", _grouped_fn=g)
        self.assertEqual(cap["channel"], "st")
        self.assertEqual(cap["query_model_name"], _KOSIMCSE)

    def test_none_resolves_active_st_bge_to_bge(self) -> None:
        # ② active='st_bge' → channel='st_bge' + BGE 질의모델 전달(질의-문서 일치).
        g, cap = _capturing_grouped()
        with mock.patch(
            "src.config.settings.get_current_settings", return_value=_fake_cfg("st_bge")
        ):
            search_hybrid("질의", _grouped_fn=g)
        self.assertEqual(cap["channel"], "st_bge")
        self.assertEqual(cap["query_model_name"], _BGE)


class TestSearchActiveChannelExplicitPrecedence(unittest.TestCase):
    """③ 명시 파라미터는 활성과 무관하게 그대로 우선(017 A/B 하니스 호환)."""

    def test_explicit_channel_and_model_override_active(self) -> None:
        # 활성이 st_bge 여도 명시 text_channel='st'·text_query_model 이 그대로 전달돼야 한다.
        g, cap = _capturing_grouped()
        with mock.patch(
            "src.config.settings.get_current_settings", return_value=_fake_cfg("st_bge")
        ):
            search_hybrid(
                "질의",
                text_channel="st",
                text_query_model="custom/model",
                _grouped_fn=g,
            )
        self.assertEqual(cap["channel"], "st")
        self.assertEqual(cap["query_model_name"], "custom/model")

    def test_explicit_channel_only_uses_channel_model_not_active(self) -> None:
        # 명시 채널만(모델 None) → 그 채널의 모델로 해소(활성 채널이 아닌 명시 채널 기준).
        g, cap = _capturing_grouped()
        with mock.patch(
            "src.config.settings.get_current_settings", return_value=_fake_cfg("st")
        ):
            search_hybrid("질의", text_channel="st_bge", _grouped_fn=g)
        self.assertEqual(cap["channel"], "st_bge")
        self.assertEqual(cap["query_model_name"], _BGE)


class TestSearchActiveChannelUninitializedFallback(unittest.TestCase):
    """④ settings 미초기화 시 기존 기본('st'/None)으로 폴백 — 006/017 회귀 가드."""

    def test_uninitialized_settings_falls_back_to_st_none(self) -> None:
        # get_current_settings 가 RuntimeError(미초기화)면 channel='st'·query_model_name=None
        # (media_search 가 기존대로 KoSimCSE 해소)으로 폴백한다. settings 없이도 동작(회귀 0).
        g, cap = _capturing_grouped()
        with mock.patch(
            "src.config.settings.get_current_settings",
            side_effect=RuntimeError("settings 미초기화"),
        ):
            search_hybrid("질의", _grouped_fn=g)
        self.assertEqual(cap["channel"], "st")
        self.assertIsNone(cap["query_model_name"])


if __name__ == "__main__":
    unittest.main()
