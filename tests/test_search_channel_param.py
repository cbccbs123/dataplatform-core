"""017 G3 — 검색 채널/질의모델 파라미터화(search_service.search_hybrid) 단위 테스트.

실제 검색(``search_media_all_grouped``)·임베딩 모델·DB 는 필요 없다 — ``_grouped_fn`` 을
스파이로 주입해 ``search_hybrid`` 가 채널·질의모델을 grouped 경로로 **어떻게 전달하는지**만
검증한다(FR-004 질의-문서 모델 일치, 회귀 0).

핵심 가드:
- **회귀 0(SC-002)**: 채널·질의모델 미지정 시 grouped 에 ``channel='st'``·``query_model_name=None``
  (=media_search 가 기존대로 KoSimCSE 로 해소)로 전달되고, 기존 인자(structured·alpha·fusion·
  include_visual 등)는 그대로 흘러간다. 기본 경로는 ``get_current_settings`` 를 건드리지 않는다
  (settings 미초기화 순수 단위에서도 동작 — 본 테스트가 settings 없이 도는 것이 그 증거).
- **A/B(FR-001)**: ``text_channel='st_bge'`` → grouped 에 그 채널 + ``model_for_channel`` 로 해소한
  BGE 질의모델이 전달된다(질의-문서 모델 일치).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from src.search.search_service import search_hybrid

# model_for_channel 이 읽는 두 필드만 가진 가짜 설정 — 실 .env·init_settings 없이 매핑을 돌린다.
_FAKE_CFG = SimpleNamespace(
    text_embedding_model="BM-K/KoSimCSE-roberta-multitask",
    text_embedding_model_bge="BAAI/bge-m3",
)


def _capturing_grouped():
    """grouped 호출 kwargs 를 잡아 두는 스파이(검색·DB 무관, 빈 버킷 반환)."""
    captured: dict[str, object] = {}

    def _g(query: str, **kw: object) -> dict[str, object]:
        captured.update(kw)
        return {"text_documents": [], "audio": [], "image": [], "video": [], "meta": {}}

    return _g, captured


class TestSearchChannelDefaultEquivalence(unittest.TestCase):
    """채널·질의모델 미지정 = 기존 동작 동치(회귀 가드)."""

    def test_default_forwards_st_channel_and_none_query_model(self) -> None:
        # 기본 호출: grouped 에 channel='st' + query_model_name=None(=KoSimCSE, media_search 가
        # 기존 cfg.text_embedding_model 로 해소). settings 미초기화 환경에서도 통과해야 한다.
        g, cap = _capturing_grouped()
        search_hybrid("질의", _grouped_fn=g)
        self.assertEqual(cap["channel"], "st")
        self.assertIsNone(cap["query_model_name"])

    def test_default_preserves_existing_grouped_kwargs(self) -> None:
        # 회귀 가드: 기존에 전달하던 인자가 그대로(같은 기본값으로) grouped 에 흘러가야 한다.
        g, cap = _capturing_grouped()
        search_hybrid("질의", _grouped_fn=g)
        self.assertEqual(cap["limit_per_bucket"], 20)
        self.assertEqual(cap["text_hybrid_alpha"], 0.75)
        self.assertEqual(cap["image_search_alpha"], 0.65)
        self.assertEqual(cap["fusion"], "alpha")
        self.assertIs(cap["include_visual"], True)
        self.assertIsNone(cap["structured"])


class TestSearchChannelBgeRouting(unittest.TestCase):
    """``text_channel='st_bge'`` → BGE 채널 + BGE 질의모델 전달(FR-004)."""

    def test_st_bge_channel_forwards_bge_query_model(self) -> None:
        g, cap = _capturing_grouped()
        # model_for_channel('st_bge') 이 활성 설정을 읽으므로 가짜 설정을 주입(순수 단위 유지).
        with mock.patch(
            "src.config.settings.get_current_settings", return_value=_FAKE_CFG
        ):
            search_hybrid("질의", text_channel="st_bge", _grouped_fn=g)
        self.assertEqual(cap["channel"], "st_bge")
        self.assertEqual(cap["query_model_name"], "BAAI/bge-m3")

    def test_unknown_channel_raises_value_error(self) -> None:
        # 미지원 채널은 잘못된 모델로 검색하지 않도록 model_for_channel 이 즉시 차단(G1).
        g, _ = _capturing_grouped()
        with mock.patch(
            "src.config.settings.get_current_settings", return_value=_FAKE_CFG
        ), self.assertRaises(ValueError):
            search_hybrid("질의", text_channel="zzz", _grouped_fn=g)


class TestSearchChannelExplicitModel(unittest.TestCase):
    """``text_query_model`` 명시 시 매핑보다 우선(설정/DB 불필요)."""

    def test_explicit_model_overrides_without_settings(self) -> None:
        # 명시 모델은 model_for_channel 을 거치지 않으므로 settings 없이도 그대로 전달된다.
        g, cap = _capturing_grouped()
        search_hybrid(
            "질의", text_channel="st", text_query_model="custom/model", _grouped_fn=g
        )
        self.assertEqual(cap["channel"], "st")
        self.assertEqual(cap["query_model_name"], "custom/model")

    def test_explicit_model_with_bge_channel_skips_mapping(self) -> None:
        # 명시 모델이 있으면 st_bge 채널이어도 model_for_channel(=settings) 을 호출하지 않는다.
        g, cap = _capturing_grouped()
        search_hybrid(
            "질의",
            text_channel="st_bge",
            text_query_model="custom/bge",
            _grouped_fn=g,
        )
        self.assertEqual(cap["channel"], "st_bge")
        self.assertEqual(cap["query_model_name"], "custom/bge")


if __name__ == "__main__":
    unittest.main()
