"""049 G2 — 캡션 v2 프롬프트 + 토글·후처리 배선 단위 테스트(FR-201/202·FR-102).

순수 단위(빌더는 LLM·settings 미접촉 순수 문자열) + summarize 토글 배선(complete_vision_json mock).

  - T201 ``test_v1_prompt_unchanged``: ``v2`` 기본 False 시 빌더 출력이 **현행 문자열과 바이트 동일**
    (FR-102 회귀 안전판 — 현재 출력을 기대값으로 고정). v2 분기를 추가해도 v1 경로를 건드리지 않음을 봉인.
  - T201 ``test_v2_prompt_has_concrete_instructions``: ``v2=True`` 출력에 구체 개체·고유명사·일반어 금지
    지시가 추가됨(FR-201/202).
  - T202 토글 배선: ``vlm_summary_prompt_v2=True`` 면 키워드를 promote_objects_to_keywords 로 후처리
    (generic 제거·objects 승격), False 면 현행 inline 루프(케이블 미승격) — complete_vision_json mock 으로 검증.
"""

from __future__ import annotations

import unittest
from unittest import mock

import src.llm.image_summarizer as image_summarizer
from src.llm.summary_postprocess import promote_objects_to_keywords

# T201: 현행 v1 캡션 프롬프트 출력(summary_max_chars=500·top_k_keywords=10). v2 분기 추가 전 캡처본 —
# v2: bool=False 기본 호출이 이 문자열과 바이트 동일해야 한다(완전 no-op·FR-102).
_V1_IMAGE_PROMPT_EXPECTED = (
    "이 이미지를 분석해서 반드시 JSON만 출력해.\n"
    "형식:\n"
    '{ "summary": "한국어로 담긴 내용·주제를 서술한 문장", '
    '"keywords": ["키워드1", "키워드2"], '
    '"objects": ["객체1", "객체2"] }\n'
    "- summary은 500자 이내\n"
    "- keywords는 핵심 키워드 최대 10개 (한국어)\n"
    "- objects는 이미지에 보이는 모든 주요 객체를 일반 명사 형태의 한국어로 나열 (중복 제거)\n"
    "- summary 는 매체 자체를 언급하지 말 것 — '이미지입니다/사진입니다/영상입니다/썸네일' 같은 "
    "매체 단어·문형 금지\n"
    "- 담긴 내용·주제·개체를 명사구 중심으로 서술\n"
    "개수/비율/합계 같은 통계 표현은 summary/keywords에는 쓰지 말 것."
)


class TestImageCaptionPromptV2(unittest.TestCase):
    def test_v1_prompt_unchanged(self) -> None:
        # FR-102: v2 기본 False → 현행 문자열과 바이트 동일(회귀 안전판).
        prompt = image_summarizer._build_image_caption_prompt(
            summary_max_chars=500, top_k_keywords=10
        )
        self.assertEqual(prompt, _V1_IMAGE_PROMPT_EXPECTED)
        # 명시적 v2=False 도 동일.
        self.assertEqual(
            image_summarizer._build_image_caption_prompt(
                summary_max_chars=500, top_k_keywords=10, v2=False
            ),
            _V1_IMAGE_PROMPT_EXPECTED,
        )

    def test_v2_prompt_has_concrete_instructions(self) -> None:
        # FR-201/202: v2 는 구체 개체·고유명사 명시 + 일반어 금지 지시를 추가한다.
        prompt = image_summarizer._build_image_caption_prompt(
            summary_max_chars=500, top_k_keywords=10, v2=True
        )
        self.assertIn("구체", prompt)
        self.assertIn("고유명사", prompt)
        self.assertTrue(
            ("일반어" in prompt) or ("금지" in prompt),
            "검색지향 키워드(일반어 금지) 지시 누락",
        )
        # v2 는 v1 토픽화 지시(매체 문형 금지)를 유지해야 한다 — v2 는 v1 위에 덧붙임.
        self.assertIn("매체", prompt)

    def test_v2_is_superset_of_v1(self) -> None:
        # v2 는 v1 프롬프트를 그대로 포함(추가 지시만 덧붙임) — v1 경로 보존 봉인.
        v2 = image_summarizer._build_image_caption_prompt(
            summary_max_chars=500, top_k_keywords=10, v2=True
        )
        self.assertIn("JSON만 출력", v2)
        self.assertIn("summary은 500자 이내", v2)


def _build_cfg_stub(*, v2: bool, top_k: int = 10, summary_max: int = 500) -> mock.Mock:
    """summarize 함수가 읽는 settings 의 필요한 필드만 가진 stub(순수 단위 — 실 settings 미초기화)."""
    cfg = mock.Mock()
    cfg.vlm_summary_prompt_v2 = v2
    cfg.top_k_keywords = top_k
    cfg.summary_max_chars = summary_max
    return cfg


class TestSummarizeImageToggleWiring(unittest.TestCase):
    """T202: vlm_summary_prompt_v2 토글에 따라 키워드 후처리 분기(complete_vision_json mock)."""

    def _fake_vision_payload(self) -> dict:
        return {"summary": "s", "keywords": ["충전기", "영상"], "objects": ["케이블"]}

    def test_summarize_v2_postprocesses(self) -> None:
        # v2: keywords == promote_objects_to_keywords(["충전기","영상"],["케이블"],limit=10) == ["충전기","케이블"]
        cfg = _build_cfg_stub(v2=True)
        with mock.patch.object(
            image_summarizer, "complete_vision_json", return_value=self._fake_vision_payload()
        ), mock.patch.object(image_summarizer, "get_current_settings", return_value=cfg):
            out = image_summarizer._summarize_image_caption_keywords_objects_from_data_url(
                image_data_url="data:image/jpeg;base64,AAAA"
            )
        self.assertEqual(
            out["keywords"],
            promote_objects_to_keywords(["충전기", "영상"], ["케이블"], limit=10),
        )
        self.assertEqual(out["keywords"], ["충전기", "케이블"])  # "영상" 제거·케이블 승격
        # objects 는 v2 에서도 보존(키워드 승격 입력일 뿐 objects 출력은 공통 코드·불변).
        self.assertEqual(out["objects"], ["케이블"])

    def test_summarize_v1_keeps_inline_dedup(self) -> None:
        # v1: 현행 inline 루프 — dedup 만(케이블 미승격·"영상" 보존).
        cfg = _build_cfg_stub(v2=False)
        with mock.patch.object(
            image_summarizer, "complete_vision_json", return_value=self._fake_vision_payload()
        ), mock.patch.object(image_summarizer, "get_current_settings", return_value=cfg):
            out = image_summarizer._summarize_image_caption_keywords_objects_from_data_url(
                image_data_url="data:image/jpeg;base64,AAAA"
            )
        self.assertEqual(out["keywords"], ["충전기", "영상"])  # 현행: dedup 만, objects 미승격
        # objects 는 v1·v2 모두 동일하게 보존(메타 후단에서 CLIP 후보로 쓰임).
        self.assertEqual(out["objects"], ["케이블"])

    def test_v1_passes_v2_false_to_builder(self) -> None:
        # 배선 검증: v1 cfg 면 빌더가 v2=False 로 호출돼 v1 프롬프트가 전달된다(complete_vision_json text 인자).
        cfg = _build_cfg_stub(v2=False)
        captured: dict = {}

        def _capture(*, text: str, image_data_url: str) -> dict:
            captured["text"] = text
            return self._fake_vision_payload()

        with mock.patch.object(image_summarizer, "complete_vision_json", side_effect=_capture), \
                mock.patch.object(image_summarizer, "get_current_settings", return_value=cfg):
            image_summarizer._summarize_image_caption_keywords_objects_from_data_url(
                image_data_url="data:image/jpeg;base64,AAAA"
            )
        self.assertEqual(captured["text"], _V1_IMAGE_PROMPT_EXPECTED)


if __name__ == "__main__":
    unittest.main()
