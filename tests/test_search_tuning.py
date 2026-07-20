"""069 US-E FR-E5② — SearchTuning 묶음·from_settings 해소·기본 폴백 봉인."""
from __future__ import annotations

import types
import unittest
from dataclasses import FrozenInstanceError, replace

from src.config import search_constants
from src.search.search_tuning import SearchTuning


def _search_ns(**over):
    """from_settings 가 읽는 cfg.search 13필드를 전부 갖춘 SimpleNamespace(기본=search_constants+override).

    PR4b: from_settings 는 완전한 cfg.search 를 요구한다(방어 getattr 폐지) — 일부만 세팅한 fake 는
    AttributeError 가 나므로 기본으로 채운 뒤 override 한다."""
    base = types.SimpleNamespace(
        fusion_weights=search_constants.OS_FUSION_WEIGHTS_DEFAULT,
        os_cutoff_enabled=search_constants.OS_CUTOFF_ENABLED_DEFAULT,
        os_cutoff_eps=search_constants.OS_CUTOFF_EPS_DEFAULT,
        os_cutoff_floor=search_constants.OS_CUTOFF_FLOOR_DEFAULT,
        os_result_floor=search_constants.OS_RESULT_FLOOR_DEFAULT,
        os_bm25_operator=search_constants.OS_BM25_OPERATOR_DEFAULT,
        os_rerank_enabled=search_constants.OS_RERANK_ENABLED_DEFAULT,
        os_rerank_top_r=search_constants.OS_RERANK_TOP_R_DEFAULT,
        os_rerank_tau=search_constants.OS_RERANK_TAU_DEFAULT,
        os_rerank_model=search_constants.OS_RERANK_MODEL_DEFAULT,
        about_filter_enabled=search_constants.SEARCH_ABOUT_FILTER_ENABLED_DEFAULT,
        evidence_rescue_enabled=search_constants.SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT,
        evidence_debug=search_constants.SEARCH_EVIDENCE_DEBUG_DEFAULT,
    )
    for k, v in over.items():
        setattr(base, k, v)
    return types.SimpleNamespace(search=base)


class TestSearchTuningDefaults(unittest.TestCase):
    """무인자 SearchTuning() = search_constants 기본(현행 동작·회귀 0)."""

    def test_defaults_match_search_constants(self) -> None:
        t = SearchTuning()
        self.assertEqual(t.weights, search_constants.OS_FUSION_WEIGHTS_DEFAULT)
        self.assertEqual(t.cutoff_enabled, search_constants.OS_CUTOFF_ENABLED_DEFAULT)
        self.assertEqual(t.cutoff_eps, search_constants.OS_CUTOFF_EPS_DEFAULT)
        self.assertEqual(t.cutoff_floor, search_constants.OS_CUTOFF_FLOOR_DEFAULT)
        self.assertEqual(t.result_floor, search_constants.OS_RESULT_FLOOR_DEFAULT)
        self.assertEqual(t.bm25_operator, search_constants.OS_BM25_OPERATOR_DEFAULT)
        self.assertEqual(t.rerank_enabled, search_constants.OS_RERANK_ENABLED_DEFAULT)
        self.assertEqual(t.rerank_top_r, search_constants.OS_RERANK_TOP_R_DEFAULT)
        self.assertEqual(t.rerank_tau, search_constants.OS_RERANK_TAU_DEFAULT)
        self.assertEqual(t.rerank_model, search_constants.OS_RERANK_MODEL_DEFAULT)
        self.assertEqual(t.about_filter_enabled, search_constants.SEARCH_ABOUT_FILTER_ENABLED_DEFAULT)
        self.assertEqual(t.evidence_rescue_enabled, search_constants.SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT)
        self.assertEqual(t.evidence_debug, search_constants.SEARCH_EVIDENCE_DEBUG_DEFAULT)

    def test_frozen(self) -> None:
        t = SearchTuning()
        with self.assertRaises(FrozenInstanceError):
            t.cutoff_enabled = False  # type: ignore[misc]


class TestFromSettings(unittest.TestCase):
    """from_settings 가 cfg 속성을 1회 해소하고, 미보유 속성은 search_constants 로 폴백한다."""

    def test_reads_cfg_values(self) -> None:
        cfg = _search_ns(
            fusion_weights=(0.3, 0.7),
            os_cutoff_enabled=True,
            os_cutoff_eps=0.22,
            os_cutoff_floor=0.55,
            os_result_floor=0.33,
            os_bm25_operator="and",
            os_rerank_enabled=True,
            os_rerank_top_r=7,
            os_rerank_tau=0.2,
            os_rerank_model="ce",
            about_filter_enabled=True,
            evidence_rescue_enabled=True,
            evidence_debug=True,
        )
        t = SearchTuning.from_settings(cfg)
        self.assertEqual(t.weights, (0.3, 0.7))
        self.assertEqual(t.cutoff_eps, 0.22)
        self.assertEqual(t.cutoff_floor, 0.55)
        self.assertEqual(t.result_floor, 0.33)
        self.assertEqual(t.bm25_operator, "and")
        self.assertTrue(t.rerank_enabled)
        self.assertEqual(t.rerank_top_r, 7)
        self.assertTrue(t.about_filter_enabled)
        self.assertTrue(t.evidence_debug)

    def test_requires_cfg_search_group(self) -> None:
        # PR4b·③: 방어 getattr 폐지 — from_settings 는 완전한 cfg.search 를 요구한다. cfg.search 미보유는
        # 조용히 기본으로 폴백하지 않고 즉시 AttributeError(fail-fast). settings 미초기화 폴백은 호출부
        # (search_service)가 cfg=None → SearchTuning() 로 명시 처리한다. 무인자 기본은 아래 별도 봉인.
        with self.assertRaises(AttributeError):
            SearchTuning.from_settings(types.SimpleNamespace())
        self.assertEqual(SearchTuning(), SearchTuning())  # 무인자 기본 = search_constants(별도 경로)

    def test_query_norm_not_in_tuning(self) -> None:
        # query_norm_enabled 은 이중 정규화 방지로 tuning 에서 의도적 제외(search_service 가 선정규화).
        self.assertNotIn("query_norm_enabled", set(SearchTuning.__dataclass_fields__))

    def test_disable_cutoff_override_via_replace(self) -> None:
        # disable_os_cutoff 디버그 우회 = replace(tuning, cutoff_enabled=False) (search_service 배선).
        cfg = _search_ns(os_cutoff_enabled=True)
        t = replace(SearchTuning.from_settings(cfg), cutoff_enabled=False)
        self.assertFalse(t.cutoff_enabled)


if __name__ == "__main__":
    unittest.main()
