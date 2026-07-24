"""044 G2 — build_search_policy / build_query_plan (2026-07-24 mode 슬림).

generic_single_term·restricted·suggestion·seed 제거 후 — mode(auto|keyword)와 content_query 만 남는다.
"""

from __future__ import annotations

import unittest

from src.search.query_plan import build_query_plan, build_search_policy, search_plan_to_meta


class BuildSearchPolicyTest(unittest.TestCase):
    def test_default_mode_auto(self) -> None:
        p = build_search_policy("무선충전기")
        self.assertEqual(p.mode, "auto")
        self.assertEqual(p.content_query, "무선충전기")

    def test_keyword_mode(self) -> None:
        p = build_search_policy("테스트", mode="keyword")
        self.assertEqual(p.mode, "keyword")

    def test_unknown_mode_falls_back_auto(self) -> None:
        self.assertEqual(build_search_policy("q", mode="bogus").mode, "auto")

    def test_no_auto_tags(self) -> None:
        # 자동 필터 승격 없음(FR-302) — content_query 는 원문 그대로.
        p = build_search_policy("테스트")
        self.assertEqual(p.content_query, "테스트")
        self.assertNotIn("tags", p.content_query)


class SearchPlanMetaTest(unittest.TestCase):
    def test_meta_has_content_query_and_mode(self) -> None:
        meta = search_plan_to_meta(build_query_plan("회식", mode="keyword"))
        self.assertEqual(meta, {"content_query": "회식", "mode": "keyword"})

    def test_meta_default_mode_auto(self) -> None:
        self.assertEqual(search_plan_to_meta(build_query_plan("회식"))["mode"], "auto")


if __name__ == "__main__":
    unittest.main()
