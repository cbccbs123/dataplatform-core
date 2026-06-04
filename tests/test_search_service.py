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

    def test_structured_forwarded_to_grouped(self) -> None:
        # structured 를 넘기면 그대로 grouped 검색에 전달돼 LLM 질의구조화를 건너뛸 수 있어야 한다.
        captured: dict[str, object] = {}

        def fake(query: str, **kw: object) -> dict[str, object]:
            captured.update(kw)
            return {"meta": {}}

        s = {"semantic_query": "재작성", "semantic_query_en": "rewritten"}
        search_hybrid("원질의", structured=s, _grouped_fn=fake)
        self.assertEqual(captured["structured"], s)


def _scored_grouped(query: str, **_kw: object) -> dict[str, object]:
    """각 버킷에 ``similarity`` 가 다른 두 행을 담은 grouped 결과(적합도 필터 검증용)."""
    return {
        "text_documents": [
            {"id": "t_hi", "similarity": 0.40},
            {"id": "t_lo", "similarity": 0.25},
        ],
        "image": [
            {"id": "i_hi", "similarity": 0.30},
            {"id": "i_lo", "similarity": 0.17},
        ],
        "audio": [{"id": "a1", "similarity": 0.10}],
        "video": [{"id": "v1", "similarity": 0.10}],
        "meta": {},
    }


class TestSearchHybridMinScore(unittest.TestCase):
    def test_filters_rows_below_threshold(self) -> None:
        out = search_hybrid(
            "질의", modalities=["text"], min_scores={"text": 0.3}, _grouped_fn=_scored_grouped
        )
        ids = [r["id"] for r in out["results"]["text_documents"]]
        self.assertEqual(ids, ["t_hi"])  # 0.25 행은 제외

    def test_none_and_zero_threshold_passthrough(self) -> None:
        out_none = search_hybrid("질의", modalities=["text"], _grouped_fn=_scored_grouped)
        out_zero = search_hybrid(
            "질의", modalities=["text"], min_scores={"text": 0.0}, _grouped_fn=_scored_grouped
        )
        self.assertEqual(len(out_none["results"]["text_documents"]), 2)
        self.assertEqual(len(out_zero["results"]["text_documents"]), 2)

    def test_threshold_is_independent_per_modality(self) -> None:
        # text 임계값이 image 버킷을 거르면 안 된다.
        out = search_hybrid(
            "질의",
            modalities=["text", "image"],
            min_scores={"text": 0.35},
            _grouped_fn=_scored_grouped,
        )
        self.assertEqual([r["id"] for r in out["results"]["text_documents"]], ["t_hi"])
        self.assertEqual(len(out["results"]["image"]), 2)  # image 임계값 미지정 → 그대로

    def test_all_below_yields_empty_bucket(self) -> None:
        out = search_hybrid(
            "질의", modalities=["image"], min_scores={"image": 0.99}, _grouped_fn=_scored_grouped
        )
        self.assertEqual(out["results"]["image"], [])

    def test_negative_threshold_disables_filter(self) -> None:
        # 음수 임계값(오타 등)은 0.0 과 동일하게 필터 비활성 — 조용히 전부 통과시킨다.
        out = search_hybrid(
            "질의", modalities=["text"], min_scores={"text": -0.3}, _grouped_fn=_scored_grouped
        )
        self.assertEqual(len(out["results"]["text_documents"]), 2)

    def test_nan_and_missing_similarity_treated_as_zero(self) -> None:
        def grouped(query: str, **_kw: object) -> dict[str, object]:
            return {
                "text_documents": [
                    {"id": "good", "similarity": 0.5},
                    {"id": "nan", "similarity": float("nan")},
                    {"id": "missing"},  # similarity 키 없음
                ],
                "meta": {},
            }

        out = search_hybrid(
            "질의", modalities=["text"], min_scores={"text": 0.1}, _grouped_fn=grouped
        )
        self.assertEqual([r["id"] for r in out["results"]["text_documents"]], ["good"])


class TestIncludeVisualWiring(unittest.TestCase):
    """요청 모달리티에 image/video 가 없으면 시각 2단계(CLIP)는 낭비이므로,
    search_hybrid 가 grouped 에 ``include_visual=False`` 를 넘겨 건너뛰게 한다(결과 동치·비용 절감).

    text/audio 만 요청 → include_visual=False. image·video 중 하나라도 포함하거나 전체(None) → True.
    """

    @staticmethod
    def _capturing_grouped():
        captured: dict[str, object] = {}

        def _g(query: str, **kw: object) -> dict[str, object]:
            captured.update(kw)
            return {"text_documents": [], "audio": [], "image": [], "video": [], "meta": {}}

        return _g, captured

    def test_text_only_skips_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["text"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], False)

    def test_text_audio_skips_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["text", "audio"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], False)

    def test_audio_only_skips_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["audio"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], False)

    def test_image_requested_keeps_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["image"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], True)

    def test_video_requested_keeps_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["video"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], True)

    def test_text_with_video_keeps_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", modalities=["text", "video"], _grouped_fn=g)
        self.assertIs(cap["include_visual"], True)

    def test_none_modalities_keeps_visual(self) -> None:
        g, cap = self._capturing_grouped()
        search_hybrid("질의", _grouped_fn=g)  # 전체 모달리티
        self.assertIs(cap["include_visual"], True)


if __name__ == "__main__":
    unittest.main()
