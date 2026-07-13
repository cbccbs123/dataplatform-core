"""006 검색 — 질의 구조화 폴백(client 주입, 네트워크 없음).

``structure_user_query`` 는 LLM 응답을 구조화 dict 로 만든다. LLM 이 스키마를 어겨
비-dict JSON(배열·스칼라)을 내도 예외 없이 안전한 기본 dict 로 폴백해야 한다(#6).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.search.query_preprocess import (
    morph_noun_phrase_query,
    noun_phrase_query,
    structure_user_query,
)


def _client_returning(content: str) -> MagicMock:
    """``complete_text`` 가 호출하는 ``chat.completions.create`` 응답을 모킹."""
    c = MagicMock()
    c.chat.completions.create.return_value.choices = [MagicMock()]
    c.chat.completions.create.return_value.choices[0].message.content = content
    return c


class TestStructureUserQueryFallback(unittest.TestCase):
    def test_valid_json_object_parsed(self) -> None:
        c = _client_returning('{"keywords": ["a"], "semantic_query": "요약"}')
        out = structure_user_query("워크숍 발표자료", client=c)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["keywords"], ["a"])
        self.assertEqual(out["semantic_query"], "요약")

    def test_json_array_falls_back_to_dict(self) -> None:
        c = _client_returning("[1, 2, 3]")
        out = structure_user_query("워크숍 발표자료", client=c)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["semantic_query"], "워크숍 발표자료")

    def test_json_scalar_falls_back_to_dict(self) -> None:
        c = _client_returning("42")
        out = structure_user_query("워크숍 발표자료", client=c)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["semantic_query"], "워크숍 발표자료")

    def test_empty_response_falls_back_to_dict(self) -> None:
        c = _client_returning("")
        out = structure_user_query("워크숍 발표자료", client=c)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["semantic_query"], "워크숍 발표자료")


class TestNounPhraseQuery(unittest.TestCase):
    """029 T007: 검색 질의 명사구 정규화(021 FR-004 토글 개정·헌법 §3 결정성 제약).

    ``complete_json`` 단일 seam 경유 ``{"query_norm":…}`` 스키마 호출·``temperature=0``(비0 리터럴 0)·
    프롬프트 env 입력 0(datetime/now/경로/랜덤 금지 — 021 비결정성 재도입 차단)·결정성·fail-safe(원문
    폴백)를 가짜 client 주입으로 네트워크 없이 봉인한다. 028 측정 근거대로 **명사구 정규화만**(풀어쓰기·
    HyDE 금지)."""

    def test_parses_query_norm_from_json_schema(self) -> None:
        # complete_json 의 {"query_norm":…} 스키마를 파싱해 명사구를 반환한다.
        c = _client_returning('{"query_norm": "천체 관측"}')
        self.assertEqual(noun_phrase_query("별 보는 방법", client=c), "천체 관측")

    def test_prompt_requests_query_norm_schema_with_query(self) -> None:
        # LLM 에 보낸 프롬프트가 query_norm 스키마 키와 사용자 질의를 담는다(complete_json 스키마 호출).
        c = _client_returning('{"query_norm": "낚시"}')
        noun_phrase_query("물고기 잡는 법", client=c)
        sent = c.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("query_norm", sent)
        self.assertIn("물고기 잡는 법", sent)

    def test_uses_json_object_response_format(self) -> None:
        # complete_text 가 아니라 complete_json 경유 — response_format=json_object 강제(평문 불가).
        c = _client_returning('{"query_norm": "낚시"}')
        noun_phrase_query("물고기 잡는 법", client=c)
        self.assertEqual(
            c.chat.completions.create.call_args.kwargs["response_format"], {"type": "json_object"}
        )

    def test_temperature_is_zero(self) -> None:
        # 헌법 §3 결정 재현성: seam 기본 temperature=0 유지(비0 리터럴 미전달 — policy_gate 차단 회피).
        c = _client_returning('{"query_norm": "낚시"}')
        noun_phrase_query("물고기 잡는 법", client=c)
        self.assertEqual(c.chat.completions.create.call_args.kwargs["temperature"], 0.0)

    def test_prompt_has_no_env_dependent_input(self) -> None:
        # 021 비결정성 재도입 차단: 프롬프트에 datetime/now/today/오늘/기준 시각/random 토큰 0
        # (순수 질의→명사구 매핑 — reference_dates_block 의 datetime.now 같은 env 입력 없음).
        c = _client_returning('{"query_norm": "낚시"}')
        noun_phrase_query("물고기 잡는 법", client=c)
        sent = c.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        for token in ("now", "today", "datetime", "timezone", "기준 시각", "오늘", "random"):
            self.assertNotIn(token, sent)

    def test_deterministic_same_query_same_phrase(self) -> None:
        # SC-003 결정성: 같은 질의 → 같은 명사구(temp=0·env 입력 0·순수 매핑).
        outs = [
            noun_phrase_query("별 보는 방법", client=_client_returning('{"query_norm": "천체 관측"}'))
            for _ in range(3)
        ]
        self.assertEqual(outs, ["천체 관측"] * 3)

    def test_empty_query_norm_falls_back_to_original(self) -> None:
        # fail-safe: query_norm 이 빈 문자열이면 원문 질의로 폴백(027 경로 — 정규화 실패가 검색을 깨지 않게).
        c = _client_returning('{"query_norm": ""}')
        self.assertEqual(noun_phrase_query("별 보는 방법", client=c), "별 보는 방법")

    def test_missing_query_norm_falls_back_to_original(self) -> None:
        c = _client_returning('{"other": "x"}')
        self.assertEqual(noun_phrase_query("별 보는 방법", client=c), "별 보는 방법")

    def test_non_string_query_norm_falls_back_to_original(self) -> None:
        c = _client_returning('{"query_norm": 42}')
        self.assertEqual(noun_phrase_query("별 보는 방법", client=c), "별 보는 방법")

    def test_empty_response_falls_back_to_original(self) -> None:
        c = _client_returning("")
        self.assertEqual(noun_phrase_query("별 보는 방법", client=c), "별 보는 방법")

    def test_empty_input_not_sent_to_llm(self) -> None:
        # 빈 질의는 정규화할 내용이 없어 LLM 미호출·원문 반환.
        c = _client_returning('{"query_norm": "X"}')
        self.assertEqual(noun_phrase_query("", client=c), "")
        c.chat.completions.create.assert_not_called()


class TestMorphNounPhraseQuery(unittest.TestCase):
    """072: 형태소 명사 정규화(analyze_fn 주입 seam·OS 없이 순수 검증). FR-001~004.

    실제 배선(nori _analyze·client)이 아니라 가짜 ``analyze_fn`` 을 주입해 판별 게이트·명사 추출·
    스톱워드 제거·폴백을 결정적으로 검증한다(헌법 3조 — 순수 함수).
    """

    NOUN = frozenset({"NNG", "NNP", "SL", "SH", "SN"})
    STOP = frozenset({"영상", "사진", "추천", "방법", "법"})

    def _analyze(self, mapping: dict[str, list[tuple[str, str]]]):
        calls: list[str] = []

        def fn(text: str) -> list[tuple[str, str]]:
            calls.append(text)
            return mapping.get(text, [])

        fn.calls = calls  # type: ignore[attr-defined]
        return fn

    def _norm(self, query, fn, min_word_tokens=3):
        return morph_noun_phrase_query(
            query, analyze_fn=fn, stopwords=self.STOP, noun_pos=self.NOUN,
            min_word_tokens=min_word_tokens,
        )

    def test_extracts_nouns_and_removes_stopwords(self) -> None:
        # 명사(NNG…)만 남기고 조사·어미(비명사)·스톱워드(법·영상·추천) 제거 → 핵심어만.
        q = "김밥 만드는 법 영상 추천"
        fn = self._analyze({q: [
            ("김밥", "NNG"), ("만들", "VV"), ("는", "ETM"),
            ("법", "NNG"), ("영상", "NNG"), ("추천", "NNG"),
        ]})
        self.assertEqual(self._norm(q, fn), "김밥")

    def test_preserves_order_and_dedupes(self) -> None:
        q = "서울 서울 여행 명소 추천"  # 서울 중복·추천 스톱워드
        fn = self._analyze({q: [
            ("서울", "NNP"), ("서울", "NNP"), ("여행", "NNG"), ("명소", "NNG"), ("추천", "NNG"),
        ]})
        self.assertEqual(self._norm(q, fn), "서울 여행 명소")

    def test_word_query_below_min_tokens_passthrough_no_analyze(self) -> None:
        # FR-001: 어절<min 이면 단어 질의 → 원문 그대로·analyze 미호출(IO·지연 0).
        fn = self._analyze({})
        self.assertEqual(self._norm("양궁", fn), "양궁")           # 1어절
        self.assertEqual(self._norm("케이팝 댄스", fn), "케이팝 댄스")  # 2어절 < 3
        self.assertEqual(fn.calls, [])  # analyze 한 번도 호출 안 됨

    def test_empty_result_falls_back_to_original(self) -> None:
        # FR-004: 남은 명사가 없으면(전부 스톱워드) 원문 폴백.
        q = "사진 영상 추천 방법"
        fn = self._analyze({q: [
            ("사진", "NNG"), ("영상", "NNG"), ("추천", "NNG"), ("방법", "NNG"),
        ]})
        self.assertEqual(self._norm(q, fn), q)

    def test_blank_query_passthrough_no_analyze(self) -> None:
        fn = self._analyze({})
        self.assertEqual(self._norm("", fn), "")
        self.assertEqual(self._norm("   ", fn), "   ")
        self.assertEqual(fn.calls, [])

    def test_deterministic_same_query_same_result(self) -> None:
        q = "겨울 한라산 설경 사진 추천"
        toks = [("겨울", "NNG"), ("한라산", "NNP"), ("설경", "NNG"), ("사진", "NNG"), ("추천", "NNG")]
        outs = [self._norm(q, self._analyze({q: toks})) for _ in range(3)]
        self.assertEqual(outs, ["겨울 한라산 설경"] * 3)


if __name__ == "__main__":
    unittest.main()
