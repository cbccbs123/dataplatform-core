"""045 Phase B — SearchFilters · OS bool.filter 변환."""

from __future__ import annotations

import unittest
from datetime import date

from src.search.search_filters import (
    SearchFilters,
    filters_to_opensearch_bool,
    parse_search_filters,
)


class FiltersToOpensearchBoolTest(unittest.TestCase):
    def test_empty_filters(self) -> None:
        self.assertEqual(filters_to_opensearch_bool(None), [])
        self.assertEqual(filters_to_opensearch_bool(SearchFilters()), [])

    def test_file_ext_clause(self) -> None:
        clauses = filters_to_opensearch_bool(SearchFilters(file_exts=("pdf", "txt")))
        self.assertEqual(clauses, [{"terms": {"filter_kw.file_ext": ["pdf", "txt"]}}])

    def test_created_at_range(self) -> None:
        clauses = filters_to_opensearch_bool(
            SearchFilters(
                created_from=date(2026, 1, 1),
                created_to=date(2026, 6, 30),
            )
        )
        self.assertEqual(
            clauses,
            [
                {"range": {"filter_date.created_at": {"gte": "2026-01-01"}}},
                {"range": {"filter_date.created_at": {"lte": "2026-06-30"}}},
            ],
        )

    def test_combined_clauses(self) -> None:
        clauses = filters_to_opensearch_bool(
            SearchFilters(file_exts=("mp3",), created_from=date(2026, 1, 1))
        )
        self.assertEqual(len(clauses), 2)


class ParseSearchFiltersTest(unittest.TestCase):
    def test_none_when_empty(self) -> None:
        self.assertIsNone(parse_search_filters())

    def test_normalizes_ext(self) -> None:
        sf = parse_search_filters(file_ext=[".PDF", "txt"])
        assert sf is not None
        self.assertEqual(sf.file_exts, ("pdf", "txt"))


if __name__ == "__main__":
    unittest.main()
