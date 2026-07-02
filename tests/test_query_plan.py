"""044 G2 — build_search_policy stub."""

from __future__ import annotations

import unittest

from src.search.query_plan import (
    build_search_policy,
    is_generic_single_term,
    merge_generic_term_seed,
)


class BuildSearchPolicyTest(unittest.TestCase):
    def test_tests_is_restricted_generic(self) -> None:
        p = build_search_policy("테스트")
        self.assertTrue(p.generic_single_term)
        self.assertEqual(p.lexical_rescue, "restricted")
        self.assertEqual(p.mode, "auto")

    def test_compound_query_is_normal(self) -> None:
        p = build_search_policy("무선충전기")
        self.assertFalse(p.generic_single_term)
        self.assertEqual(p.lexical_rescue, "normal")

    def test_no_auto_tags_test(self) -> None:
        p = build_search_policy("테스트")
        self.assertEqual(p.content_query, "테스트")
        self.assertNotIn("tags", p.content_query)

    def test_keyword_mode(self) -> None:
        p = build_search_policy("테스트", mode="keyword")
        self.assertEqual(p.mode, "keyword")
        self.assertEqual(p.lexical_rescue, "normal")

    def test_seed_casefold(self) -> None:
        self.assertTrue(is_generic_single_term("TEST"))


class QueryPlanSuggestionsTest(unittest.TestCase):
    def test_tests_has_keyword_suggestion(self) -> None:
        from src.search.query_plan import build_query_plan

        plan = build_query_plan("테스트")
        self.assertTrue(plan.policy.generic_single_term)
        self.assertGreater(len(plan.suggestions), 0)
        self.assertIn("keyword", plan.suggestions[0])

    def test_compound_no_suggestions(self) -> None:
        from src.search.query_plan import build_query_plan

        plan = build_query_plan("무선충전기")
        self.assertEqual(plan.suggestions, ())


class SearchPlanMetaAudienceTest(unittest.TestCase):
    """057 FR-203 — 포털 응답 suggestions 를 {text, audience} 로 태깅(프론트 정규식 대체).

    프론트가 정규식(isDevFacingSuggestion)으로 "mode=keyword"·"단어 포함" 류 dev 힌트를 거르던 판정을
    서버가 audience("user"|"dev")로 넘긴다. 생성 출처(_build_suggestions)가 아는 권위 분류라 정규식 표류가
    없다. mode=keyword 안내는 API 파라미터 조작 지시라 dev-facing.
    """

    def test_meta_suggestions_are_audience_tagged_objects(self) -> None:
        from src.search.query_plan import build_query_plan, search_plan_to_meta

        meta = search_plan_to_meta(build_query_plan("테스트"))
        # 문자열 리스트가 아니라 {text, audience} 객체 리스트여야 한다(프론트 정규식 제거 근거).
        self.assertIn("suggestions", meta)
        self.assertEqual(len(meta["suggestions"]), 1)
        s = meta["suggestions"][0]
        self.assertIsInstance(s, dict)
        self.assertIn("keyword", s["text"])
        self.assertEqual(s["audience"], "dev")  # mode=keyword 안내 = dev-facing

    def test_meta_no_suggestions_key_when_empty(self) -> None:
        from src.search.query_plan import build_query_plan, search_plan_to_meta

        # 복합어(제안 없음)면 suggestions 키 자체가 없다(기존 minimal 노출 규칙 보존).
        meta = search_plan_to_meta(build_query_plan("무선충전기"))
        self.assertNotIn("suggestions", meta)

    def test_unmapped_suggestion_defaults_to_user(self) -> None:
        # 매핑에 없는 신규 제안은 보수적으로 user(표시) — 현행 프론트 동작(dev 만 거름)과 정합.
        from src.search.query_plan import _suggestion_audience

        self.assertEqual(_suggestion_audience("아무 사용자 안내 문구"), "user")

    def test_keyword_mode_suggestion_mapped_dev(self) -> None:
        from src.search.query_plan import _KEYWORD_MODE_SUGGESTION, _suggestion_audience

        self.assertEqual(_suggestion_audience(_KEYWORD_MODE_SUGGESTION), "dev")


class MergeGenericTermSeedTest(unittest.TestCase):
    def test_dedup_casefold(self) -> None:
        merged = merge_generic_term_seed(("test",), ("TEST", "foo"))
        self.assertEqual(merged, ("test", "foo"))

    def test_extra_seed_is_generic(self) -> None:
        seed = merge_generic_term_seed((), ("foo", "bar"))
        self.assertTrue(is_generic_single_term("foo", seed=seed))
        self.assertTrue(is_generic_single_term("bar", seed=seed))
        self.assertFalse(is_generic_single_term("foobar", seed=seed))


if __name__ == "__main__":
    unittest.main()
