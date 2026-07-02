"""056 G6 — 주제 필터(topic/subtopic terms) 단위 테스트 (FR-503).

전략
    ``SearchFilters`` 에 추가된 ``topic``/``subtopic`` 필드와 ``parse_search_filters`` 파라미터,
    그리고 ``filters_to_opensearch_bool`` → OS ``terms`` 절 변환, ``build_bm25_body``/
    ``build_knn_body`` 의 bool ``filter`` 절 삽입을 순수 단위로 단언한다(DB·OS·LLM 0).

    **철회 반영(2026-07-02)**: topics_text(BM25 보강) 필드는 스코프 철회 — 주제는 keyword
    ``terms`` **필터**로만 검색에 반영한다. terms 필터는 결정적이라 랭킹에 영향이 없다. 따라서
    본 테스트는 ``topics_text^boost``/``cross_fields`` 합류를 단언하지 않고, 오히려 **should
    (랭킹) 절이 topic 필터 유무와 무관하게 동일**함을 단언해 랭킹 무영향을 봉인한다.
"""
from __future__ import annotations

import unittest

from src.search.opensearch_search import build_bm25_body, build_knn_body
from src.search.search_filters import (
    SearchFilters,
    filters_to_opensearch_bool,
    parse_search_filters,
)


class ParseTopicFilterTest(unittest.TestCase):
    """parse_search_filters(topic=…/subtopic=…) → SearchFilters.topic/subtopic."""

    def test_topic_only(self) -> None:
        sf = parse_search_filters(topic="요리")
        assert sf is not None
        self.assertEqual(sf.topic, "요리")
        self.assertIsNone(sf.subtopic)

    def test_subtopic_only(self) -> None:
        sf = parse_search_filters(subtopic="제빵")
        assert sf is not None
        self.assertEqual(sf.subtopic, "제빵")
        self.assertIsNone(sf.topic)

    def test_topic_and_subtopic(self) -> None:
        sf = parse_search_filters(topic="요리", subtopic="제빵")
        assert sf is not None
        self.assertEqual(sf.topic, "요리")
        self.assertEqual(sf.subtopic, "제빵")

    def test_blank_topic_is_ignored(self) -> None:
        # 공백만인 topic 은 필터 비활성 → 다른 필터도 없으면 None.
        self.assertIsNone(parse_search_filters(topic="   "))

    def test_topic_not_casefolded(self) -> None:
        # 주제는 색인된 keyword 원문과 정확 일치해야 하므로 casefold/정규화하지 않는다(strip 만).
        sf = parse_search_filters(topic="  무선충전  ")
        assert sf is not None
        self.assertEqual(sf.topic, "무선충전")


class TopicFilterToOpensearchTest(unittest.TestCase):
    """filters_to_opensearch_bool → topics/subtopics terms 절."""

    def test_topic_terms_clause(self) -> None:
        clauses = filters_to_opensearch_bool(SearchFilters(topic="요리"))
        self.assertIn({"terms": {"topics": ["요리"]}}, clauses)

    def test_subtopic_terms_clause(self) -> None:
        clauses = filters_to_opensearch_bool(SearchFilters(subtopic="제빵"))
        self.assertIn({"terms": {"subtopics": ["제빵"]}}, clauses)

    def test_topic_and_subtopic_both_present(self) -> None:
        clauses = filters_to_opensearch_bool(SearchFilters(topic="요리", subtopic="제빵"))
        self.assertIn({"terms": {"topics": ["요리"]}}, clauses)
        self.assertIn({"terms": {"subtopics": ["제빵"]}}, clauses)

    def test_no_topic_no_clause(self) -> None:
        self.assertEqual(filters_to_opensearch_bool(SearchFilters()), [])


class BuildBodyTopicFilterTest(unittest.TestCase):
    """build_bm25_body/build_knn_body 가 topic 필터를 bool.filter 절에 넣는다(랭킹 무영향)."""

    def test_bm25_topic_in_filter_clause(self) -> None:
        body = build_bm25_body(
            "레시피", modality_values=["text"], k=10,
            search_filters=SearchFilters(topic="요리"),
        )
        filters = body["query"]["bool"]["filter"]
        self.assertIn({"terms": {"topics": ["요리"]}}, filters)

    def test_bm25_subtopic_in_filter_clause(self) -> None:
        body = build_bm25_body(
            "레시피", modality_values=["text"], k=10,
            search_filters=SearchFilters(subtopic="제빵"),
        )
        filters = body["query"]["bool"]["filter"]
        self.assertIn({"terms": {"subtopics": ["제빵"]}}, filters)

    def test_bm25_topic_does_not_change_ranking(self) -> None:
        # 결정적 filter 라 랭킹(should) 절은 topic 필터 유무와 무관하게 동일해야 한다(철회·무영향).
        base = build_bm25_body("레시피", modality_values=["text"], k=10)
        with_topic = build_bm25_body(
            "레시피", modality_values=["text"], k=10,
            search_filters=SearchFilters(topic="요리"),
        )
        self.assertEqual(
            with_topic["query"]["bool"]["should"], base["query"]["bool"]["should"]
        )

    def test_knn_topic_in_native_filter(self) -> None:
        body = build_knn_body(
            [0.1, 0.2, 0.3], modality_values=["text"], k=10,
            search_filters=SearchFilters(topic="요리"),
        )
        native_filter = body["query"]["knn"]["embedding"]["filter"]["bool"]["filter"]
        self.assertIn({"terms": {"topics": ["요리"]}}, native_filter)


if __name__ == "__main__":
    unittest.main()
