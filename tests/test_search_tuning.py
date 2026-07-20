"""069 US-E FR-E5② — SearchTuning 묶음·from_settings 해소·기본 폴백 봉인."""
from __future__ import annotations

import types
import unittest
from dataclasses import FrozenInstanceError, replace

from src.config import search_constants
from src.search.search_tuning import SearchTuning


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
        cfg = types.SimpleNamespace(
            opensearch_fusion_weights=(0.3, 0.7),
            search_os_cutoff_enabled=True,
            search_os_cutoff_eps=0.22,
            search_os_cutoff_floor=0.55,
            search_os_result_floor=0.33,
            search_os_bm25_operator="and",
            search_os_rerank_enabled=True,
            search_os_rerank_top_r=7,
            search_os_rerank_tau=0.2,
            search_os_rerank_model="ce",
            search_about_filter_enabled=True,
            search_evidence_rescue_enabled=True,
            search_evidence_debug=True,
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

    def test_missing_attrs_fall_back_to_constants(self) -> None:
        # 빈 cfg(속성 0) → 전 필드 search_constants 기본(settings 미초기화 순수 단위 방어).
        t = SearchTuning.from_settings(types.SimpleNamespace())
        self.assertEqual(t, SearchTuning())

    def test_query_norm_not_in_tuning(self) -> None:
        # query_norm_enabled 은 이중 정규화 방지로 tuning 에서 의도적 제외(search_service 가 선정규화).
        self.assertNotIn("query_norm_enabled", set(SearchTuning.__dataclass_fields__))

    def test_disable_cutoff_override_via_replace(self) -> None:
        # disable_os_cutoff 디버그 우회 = replace(tuning, cutoff_enabled=False) (search_service 배선).
        cfg = types.SimpleNamespace(search_os_cutoff_enabled=True)
        t = replace(SearchTuning.from_settings(cfg), cutoff_enabled=False)
        self.assertFalse(t.cutoff_enabled)


if __name__ == "__main__":
    unittest.main()
