"""069 US-E FR-E5① — opensearch_search 3분할의 모듈 경계·재export seam 봉인.

간접 테스트(opensearch_search 재export 경유)는 이미 test_opensearch_search* 가 커버한다. 여기서는
분할의 **핵심 주장 2가지**를 직접 고정한다(재export 가 나중에 실수로 끊겨도 즉시 탐지):
  (1) query_builder·fusion 이 **직접 import·호출** 가능한 순수 모듈이다(OS·opensearch-py 없이).
  (2) opensearch_search.<name> 은 두 신규 모듈의 **동일 객체**를 재export 한다(patch seam·import 보존).
"""
from __future__ import annotations

import unittest

from src.search import fusion, opensearch_search, query_builder


class TestPureModulesDirectImport(unittest.TestCase):
    """query_builder·fusion 은 OS 없이 직접 import·호출되는 순수 함수 모듈이다(FR-E5①)."""

    def test_query_builder_builds_bm25_body(self) -> None:
        body = query_builder.build_bm25_body("등산", modality_values={"text"}, k=5)
        self.assertEqual(body["size"], 5)
        self.assertIn("bool", body["query"])
        self.assertEqual(body["query"]["bool"]["minimum_should_match"], 1)

    def test_query_builder_builds_knn_body(self) -> None:
        body = query_builder.build_knn_body([0.1, 0.2], modality_values={"text"}, k=3)
        self.assertEqual(body["size"], 3)
        self.assertIn("knn", body["query"])

    def test_fusion_pure_functions(self) -> None:
        # 순수 융합/게이트/컷 수학 — OS·opensearch-py 미접촉으로 결정적 동작.
        self.assertEqual(fusion.fuse_hybrid([], [], weights=(0.5, 0.5)), [])
        self.assertEqual(fusion.minmax_normalize([]), [])
        self.assertEqual(fusion.gate_signal([]), (0.0, 0.0))
        self.assertTrue(fusion.passes_cutoff(0.9, 0.1, eps=0.1, floor=0.5))
        self.assertFalse(fusion.passes_cutoff(0.4, 0.1, eps=0.1, floor=0.5))


class TestReexportIdentity(unittest.TestCase):
    """opensearch_search 는 두 신규 모듈의 **동일 객체**를 재export 한다(patch seam·하위호환 봉인)."""

    def test_fusion_symbols_are_reexported_identically(self) -> None:
        for name in (
            "fuse_hybrid", "gate_signal", "cut_rows", "passes_cutoff",
            "rerank_reorder", "normalize_query", "os_hit_to_row",
            "knn_score_to_cosine", "minmax_normalize",
        ):
            self.assertIs(
                getattr(opensearch_search, name), getattr(fusion, name),
                f"{name} 재export 가 fusion 원본과 다른 객체 — patch seam 파손",
            )

    def test_query_builder_symbols_are_reexported_identically(self) -> None:
        for name in (
            "build_bm25_body", "build_knn_body", "BM25_NAMED_QUERY_NAMES",
            "_MODALITY_VALUES", "_lexical_clause",
        ):
            self.assertIs(
                getattr(opensearch_search, name), getattr(query_builder, name),
                f"{name} 재export 가 query_builder 원본과 다른 객체 — patch seam 파손",
            )

    def test_all_declares_reexports(self) -> None:
        # __all__ 등재로 외부 import/patch 표면을 명시(F401 억제 + 하위호환 계약).
        for name in ("fuse_hybrid", "build_bm25_body", "search_assets_os", "embed_query"):
            self.assertIn(name, opensearch_search.__all__)

    def test_search_assets_os_calls_module_globals(self) -> None:
        # search_assets_os 본문의 fuse_hybrid(...) 등이 로컬 import 로 가려지지 않고 모듈 전역을
        # 참조해야 opensearch_search.<name> monkeypatch 가 적용된다(patch seam 핵심).
        self.assertIs(opensearch_search.search_assets_os.__globals__, opensearch_search.__dict__)


if __name__ == "__main__":
    unittest.main()
