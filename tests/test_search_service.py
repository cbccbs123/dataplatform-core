"""006 검색 서비스 진입점(search_service.search_hybrid) — 호출부 독립 함수의 단위 테스트.

실제 검색(``search_media_all_grouped``)은 DB·모델이 필요하므로 ``_grouped_fn`` 주입으로
서비스 계층의 모달리티 필터·응답 모양만 네트워크 없이 검증한다(실 동작은 T016 e2e).
"""

from __future__ import annotations

import unittest

from src.search.search_service import search_hybrid


def _fake_grouped(query: str, **_kw: object) -> dict[str, object]:
    return {
        "text_documents": [{"id": "t1"}],
        "audio": [{"id": "a1"}],
        "image": [{"id": "i1"}],
        "video": [{"id": "v1"}],
        "meta": {"structured": {"semantic_query": query}},
    }


class TestSearchHybridService(unittest.TestCase):
    def test_returns_all_buckets_by_default(self) -> None:
        out = search_hybrid("질의", _grouped_fn=_fake_grouped)
        self.assertEqual(
            set(out["results"].keys()),
            {"text_documents", "audio", "image", "video"},
        )
        self.assertEqual(out["query"], "질의")

    def test_filters_to_requested_modalities(self) -> None:
        out = search_hybrid("질의", modalities=["text", "image"], _grouped_fn=_fake_grouped)
        self.assertEqual(set(out["results"].keys()), {"text_documents", "image"})
        self.assertEqual(out["results"]["image"], [{"id": "i1"}])

    def test_unknown_modality_raises(self) -> None:
        with self.assertRaises(ValueError):
            search_hybrid("질의", modalities=["bogus"], _grouped_fn=_fake_grouped)

    def test_meta_passthrough(self) -> None:
        out = search_hybrid("질의", _grouped_fn=_fake_grouped)
        self.assertEqual(out["meta"]["structured"]["semantic_query"], "질의")


if __name__ == "__main__":
    unittest.main()
