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
