"""049 G3 — reduce v2 프롬프트 + 토글·후처리 배선 단위 테스트(FR-301/302·FR-102).

순수 단위(빌더는 LLM·settings 미접촉 순수 문자열) + summarize 토글 배선(complete_json mock).

  - T301 ``test_v1_reduce_prompt_unchanged``: ``v2`` 기본 False 시 빌더 출력이 **현행 문자열과 바이트
    동일**(FR-102 회귀 안전판 — 현재 출력을 기대값으로 고정).
  - T301 ``test_v2_reduce_has_synthesis_instructions``: ``v2=True`` 출력에 영상 전체 주제·하위 주제·통합
    지시가 추가됨(FR-301/302).
  - T302 토글 배선: ``vlm_summary_prompt_v2=True`` 면 키워드를 promote_objects_to_keywords 로 후처리,
    False 면 현행 inline 루프 — complete_json mock 으로 검증.
"""

from __future__ import annotations

import unittest
from unittest import mock

import src.llm.video_summarizer as video_summarizer
from src.llm.summary_postprocess import promote_objects_to_keywords

# T301: 현행 v1 reduce 프롬프트의 머리말(타임라인 앞 본문). v2: bool=False 기본이 이 본문 + 동일한
# 장면 결과 꼬리를 바이트 동일하게 만들어야 한다(FR-102). 장면 결과 꼬리는 timeline_lines 로 동적이라
# 테스트에서 동일 lines 를 넣어 전체 문자열을 직접 비교한다.
_SCENE_LINE = "[scene 0] 0.00s~1.00s (대표 0.50s): summary=무선 충전기 | keywords=충전기 | objects=충전기"

_V1_VIDEO_PROMPT_EXPECTED = (
    "아래는 영상의 장면별 대표프레임 분석 결과다. 전체를 종합해 반드시 JSON만 출력해.\n"
    "형식:\n"
    '{ "summary": "내용·주제 중심 요약", "keywords": ["키워드1"], "objects": ["객체1"] }\n'
    "- summary는 500자 이내\n"
    "- keywords는 핵심 키워드 최대 10개 (한국어)\n"
    "- objects는 영상 전반의 주요 객체를 일반 명사 형태 한국어로 중복 없이 나열\n"
    "- summary 는 매체 자체를 언급하지 말 것 — '이미지입니다/사진입니다/영상입니다/썸네일' 같은 "
    "매체 단어·문형 금지\n"
    "- 담긴 내용·주제·개체를 명사구 중심으로 서술\n"
    "장면 순서를 반영해 흐름 중심으로 요약하고, 개수/비율/합계 같은 통계 표현은 금지.\n\n"
    "장면 결과:\n"
    + _SCENE_LINE
)


class TestVideoReducePromptV2(unittest.TestCase):
    def test_v1_reduce_prompt_unchanged(self) -> None:
        # FR-102: v2 기본 False → 현행 문자열과 바이트 동일(회귀 안전판).
        prompt = video_summarizer._build_video_summary_prompt(
            [_SCENE_LINE], summary_max_chars=500, top_k_keywords=10
        )
        self.assertEqual(prompt, _V1_VIDEO_PROMPT_EXPECTED)
        self.assertEqual(
            video_summarizer._build_video_summary_prompt(
                [_SCENE_LINE], summary_max_chars=500, top_k_keywords=10, v2=False
            ),
            _V1_VIDEO_PROMPT_EXPECTED,
        )

    def test_v2_reduce_has_synthesis_instructions(self) -> None:
        # FR-301/302: v2 는 장면 나열 대신 전체 주제 + 하위 주제 종합 + distinctive 키워드 통합 지시.
        prompt = video_summarizer._build_video_summary_prompt(
            [_SCENE_LINE], summary_max_chars=500, top_k_keywords=10, v2=True
        )
        self.assertIn("전체 주제", prompt)
        self.assertIn("하위 주제", prompt)
        self.assertIn("통합", prompt)
        # v2 도 v1 토픽화 지시(매체 문형 금지)·장면 결과 꼬리를 유지한다.
        self.assertIn("매체", prompt)
        self.assertIn("장면 결과:", prompt)
        self.assertIn(_SCENE_LINE, prompt)


def _build_cfg_stub(*, v2: bool, top_k: int = 10, summary_max: int = 500) -> mock.Mock:
    cfg = mock.Mock()
    cfg.vlm.summary_prompt_v2 = v2
    cfg.top_k_keywords = top_k
    cfg.summary_max_chars = summary_max
    return cfg


def _scene_results() -> list[dict]:
    return [
        {
            "scene_index": 0,
            "start_sec": 0.0,
            "end_sec": 1.0,
            "frame_sec": 0.5,
            "summary": {"summary": "무선 충전기", "keywords": ["충전기"], "objects": ["케이블"]},
        }
    ]


class TestSummarizeVideoToggleWiring(unittest.TestCase):
    """T302: vlm_summary_prompt_v2 토글에 따라 키워드 후처리 분기(complete_json mock)."""

    def _payload(self) -> dict:
        return {"summary": "s", "keywords": ["충전기", "영상"], "objects": ["케이블"]}

    def test_summarize_video_v2_postprocesses(self) -> None:
        # v2: keywords == promote_objects_to_keywords(["충전기","영상"],["케이블"],limit=10) == ["충전기","케이블"]
        cfg = _build_cfg_stub(v2=True)
        with mock.patch.object(
            video_summarizer, "complete_json", return_value=self._payload()
        ), mock.patch.object(video_summarizer, "get_current_settings", return_value=cfg):
            out = video_summarizer.summarize_video_from_scene_results(_scene_results())
        self.assertEqual(
            out["keywords"],
            promote_objects_to_keywords(["충전기", "영상"], ["케이블"], limit=10),
        )
        self.assertEqual(out["keywords"], ["충전기", "케이블"])  # "영상" 제거·케이블 승격

    def test_summarize_video_v1_keeps_inline_dedup(self) -> None:
        # v1: 현행 inline 루프 — dedup 만(케이블 미승격·"영상" 보존).
        cfg = _build_cfg_stub(v2=False)
        with mock.patch.object(
            video_summarizer, "complete_json", return_value=self._payload()
        ), mock.patch.object(video_summarizer, "get_current_settings", return_value=cfg):
            out = video_summarizer.summarize_video_from_scene_results(_scene_results())
        self.assertEqual(out["keywords"], ["충전기", "영상"])
        self.assertEqual(out["objects"], ["케이블"])

    def test_v1_passes_v2_false_to_builder(self) -> None:
        # 배선 검증: v1 cfg 면 빌더가 v2=False 로 호출돼 v1 reduce 프롬프트가 complete_json 에 전달된다.
        cfg = _build_cfg_stub(v2=False)
        captured: dict = {}

        def _capture(prompt: str) -> dict:
            captured["prompt"] = prompt
            return self._payload()

        with mock.patch.object(video_summarizer, "complete_json", side_effect=_capture), \
                mock.patch.object(video_summarizer, "get_current_settings", return_value=cfg):
            video_summarizer.summarize_video_from_scene_results(_scene_results())
        # v1 머리말(매체 문형 금지 지시·장면 결과 헤더)이 그대로 들어가고, v2 종합 지시는 없다.
        self.assertIn("장면 순서를 반영해 흐름 중심으로 요약", captured["prompt"])
        self.assertNotIn("전체 주제", captured["prompt"])

    def test_v2_passes_v2_true_to_builder(self) -> None:
        # 배선 검증: v2 cfg 면 빌더가 v2=True 로 호출돼 종합 지시가 complete_json 프롬프트에 들어간다.
        cfg = _build_cfg_stub(v2=True)
        captured: dict = {}

        def _capture(prompt: str) -> dict:
            captured["prompt"] = prompt
            return self._payload()

        with mock.patch.object(video_summarizer, "complete_json", side_effect=_capture), \
                mock.patch.object(video_summarizer, "get_current_settings", return_value=cfg):
            video_summarizer.summarize_video_from_scene_results(_scene_results())
        self.assertIn("전체 주제", captured["prompt"])
        self.assertIn("통합", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
