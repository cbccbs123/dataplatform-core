"""026 G1 (T001) — 요약·캡션 프롬프트 토픽화 가드(FR-001).

image/video/text summarizer 의 프롬프트가
  ① 매체 서술 금지: '이미지/사진/영상/썸네일' 같은 매체 단어·'~입니다' 문형을 쓰지 말라는 지시,
  ② 내용·주제·개체 중심(명사구) 서술 지시
를 포함하는지 단언한다. 산출 JSON 스키마(summary/keywords/objects)·파싱 로직은 **불변**임을 함께
봉인한다(temperature=0·헌법 3조 — 프롬프트 문구만 바꾸고 키·파싱은 건드리지 않는다).

프롬프트는 LLM·settings 없이 검증 가능한 **순수 빌더 함수**로 추출돼 있어 OS/DB/LLM 불필요(순수 단위).
프롬프트 상수가 함수 내 f-string 으로 숨어 있으면 가드할 수 없으므로, 각 summarizer 는 프롬프트를
순수 빌더(`_build_*_prompt`)로 노출한다(동작 불변 — 빌더 출력이 그대로 LLM 입력).
"""

from __future__ import annotations

import unittest

import src.llm.image_summarizer as image_summarizer
import src.llm.text_summarizer as text_summarizer
import src.llm.video_summarizer as video_summarizer

# 매체 서술 금지 지시의 핵심 문구(프롬프트에 반드시 포함). '매체'(개념)·'언급하지'(금지 동사) +
# 금지 매체 단어 예시 4종 — 프롬프트가 매체 문형을 명시적으로 막고 있음을 봉인한다.
_PROHIBITION_PHRASES = ("매체", "언급하지")
_MEDIA_WORDS = ("이미지", "사진", "영상", "썸네일")
# 내용·주제 중심 지시 — '명사구' + (내용/주제/개체 중 하나 이상).
_CONTENT_PHRASE = "명사구"
_CONTENT_FOCUS_WORDS = ("내용", "주제", "개체")


def _assert_topic_instruction(test: unittest.TestCase, prompt: str) -> None:
    """프롬프트가 매체 서술 금지 + 내용 중심 지시를 모두 담았는지 단언."""
    for phrase in _PROHIBITION_PHRASES:
        test.assertIn(phrase, prompt, f"매체 서술 금지 지시 '{phrase}' 누락")
    for word in _MEDIA_WORDS:
        test.assertIn(word, prompt, f"금지 매체 단어 예시 '{word}' 누락")
    test.assertIn(_CONTENT_PHRASE, prompt, "내용 중심('명사구') 지시 누락")
    test.assertTrue(
        any(w in prompt for w in _CONTENT_FOCUS_WORDS),
        "내용·주제·개체 중심 서술 지시 누락",
    )


class ImagePromptTopicTest(unittest.TestCase):
    def test_caption_prompt_has_topic_instruction(self) -> None:
        prompt = image_summarizer._build_image_caption_prompt(
            summary_max_chars=500, top_k_keywords=10
        )
        _assert_topic_instruction(self, prompt)

    def test_json_schema_keys_unchanged(self) -> None:
        # 산출 스키마 불변(FR-001): 프롬프트 JSON 형식에 summary/keywords/objects 키가 그대로 있고,
        # 결과 TypedDict 키도 동일 — 파싱 로직이 의존하는 계약이 깨지지 않았음을 봉인.
        prompt = image_summarizer._build_image_caption_prompt(
            summary_max_chars=500, top_k_keywords=10
        )
        for key in ("summary", "keywords", "objects"):
            self.assertIn(key, prompt)
        self.assertEqual(
            set(image_summarizer.ImageSummaryResult.__annotations__),
            {"summary", "keywords", "objects"},
        )


class VideoPromptTopicTest(unittest.TestCase):
    def test_summary_prompt_has_topic_instruction(self) -> None:
        prompt = video_summarizer._build_video_summary_prompt(
            ["[scene 0] 0.00s~1.00s (대표 0.50s): summary=무선 충전기 | keywords=충전기 | objects=충전기"],
            summary_max_chars=500,
            top_k_keywords=10,
        )
        _assert_topic_instruction(self, prompt)

    def test_json_schema_keys_unchanged(self) -> None:
        prompt = video_summarizer._build_video_summary_prompt(
            [], summary_max_chars=500, top_k_keywords=10
        )
        for key in ("summary", "keywords", "objects"):
            self.assertIn(key, prompt)
        self.assertEqual(
            set(video_summarizer.VideoSummaryResult.__annotations__),
            {"summary", "keywords", "objects"},
        )


class TextPromptTopicTest(unittest.TestCase):
    def test_chunk_prompt_has_topic_instruction(self) -> None:
        prompt = text_summarizer._build_chunk_summary_prompt(
            "본문 텍스트", summary_max_chars=500
        )
        _assert_topic_instruction(self, prompt)

    def test_reduce_prompt_has_topic_instruction(self) -> None:
        prompt = text_summarizer._build_reduce_prompt(
            "- 부분요약1\n- 부분요약2", summary_max_chars=500, top_k_keywords=10
        )
        _assert_topic_instruction(self, prompt)

    def test_audio_prompt_has_topic_instruction(self) -> None:
        prompt = text_summarizer._build_audio_prompt(
            "STT 전사 텍스트", summary_max_chars=500, top_k_keywords=10
        )
        _assert_topic_instruction(self, prompt)

    def test_audio_prompt_first_sentence_is_stt_specific(self) -> None:
        # 069 B6(P2-6): _build_audio_prompt 첫 문장이 reduce("청크별 요약 목록") 복붙이면
        # 오디오 summary 품질이 검색까지 전파되는 결함 — STT 전용 문구로 교체됨을 봉인한다.
        prompt = text_summarizer._build_audio_prompt(
            "STT 전사 텍스트", summary_max_chars=500, top_k_keywords=10
        )
        self.assertIn("오디오 음성 전사(STT) 전문 요약", prompt)
        self.assertNotIn("청크별 요약 목록", prompt)

    def test_json_schema_keys_unchanged(self) -> None:
        # 텍스트/오디오 산출 스키마(summary/keywords) 불변. 오디오는 stt 필드도 보존.
        chunk = text_summarizer._build_chunk_summary_prompt("본문", summary_max_chars=500)
        self.assertIn("summary", chunk)
        reduce = text_summarizer._build_reduce_prompt("m", summary_max_chars=500, top_k_keywords=10)
        for key in ("summary", "keywords"):
            self.assertIn(key, reduce)
        self.assertEqual(
            set(text_summarizer.SummaryKeywords.__annotations__),
            {"summary", "keywords", "stt"},
        )


if __name__ == "__main__":
    unittest.main()
