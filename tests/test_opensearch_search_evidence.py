"""044 G1/G2 — named BM25 · matched_queries · evidence rescue."""

from __future__ import annotations

import unittest

from src.search.opensearch_search import (
    BM25_NAMED_QUERY_NAMES,
    build_bm25_body,
    fuse_hybrid,
    os_hit_to_row,
    search_assets_os,
)
from tests.test_opensearch_search import _FakeMsearchClient, _bm25_hit_os, _knn_hit_os


def _collect_named_names(body: dict) -> set[str]:
    names: set[str] = set()
    for clause in body["query"]["bool"]["should"]:
        if "match" in clause:
            for inner in clause["match"].values():
                names.add(inner["_name"])
        elif "term" in clause:
            for inner in clause["term"].values():
                names.add(inner["_name"])
        elif "multi_match" in clause:
            names.add(clause["multi_match"]["_name"])
    return names


def _first_match_query(body: dict) -> str:
    for clause in body["query"]["bool"]["should"]:
        if "match" in clause:
            for inner in clause["match"].values():
                return str(inner["query"])
    raise AssertionError("match 절 없음")


class BuildBm25NamedQueryTest(unittest.TestCase):
    def test_build_bm25_body_named_queries(self) -> None:
        body = build_bm25_body("테스트", modality_values=["txt"], k=10, operator="and")
        names = _collect_named_names(body)
        self.assertEqual(names, set(BM25_NAMED_QUERY_NAMES))
        self.assertEqual(body["query"]["bool"]["minimum_should_match"], 1)
        cross = [
            c["multi_match"]
            for c in body["query"]["bool"]["should"]
            if "multi_match" in c
        ]
        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0]["_name"], "hit_cross_meta")
        self.assertEqual(cross[0]["type"], "cross_fields")

    def test_operator_and_on_match_clauses(self) -> None:
        body = build_bm25_body("한국어 검색", modality_values=["txt"], k=10, operator="and")
        for clause in body["query"]["bool"]["should"]:
            if "match" in clause:
                for inner in clause["match"].values():
                    self.assertEqual(inner.get("operator"), "and")
            elif "multi_match" in clause:
                self.assertEqual(clause["multi_match"].get("operator"), "and")

    def test_operator_or_omits_key(self) -> None:
        body = build_bm25_body("q", modality_values=["txt"], k=10, operator="or")
        for clause in body["query"]["bool"]["should"]:
            if "match" in clause:
                for inner in clause["match"].values():
                    self.assertNotIn("operator", inner)


class MatchedQueriesPreserveTest(unittest.TestCase):
    def test_os_hit_to_row_preserves_matched_queries(self) -> None:
        hit = {
            "_id": "a1",
            "_score": 1.0,
            "matched_queries": ["hit_summary", "hit_cross_meta"],
            "_source": {"asset_id": "a1", "summary": "충전 테스트"},
        }
        row = os_hit_to_row(hit)
        self.assertEqual(row["matched_queries"], ["hit_summary", "hit_cross_meta"])

    def test_fuse_hybrid_phase1_policy_unchanged(self) -> None:
        # gate_fail + bm25-only weak — Phase 1/G1: legacy lexical rescue 유지(전부 keep).
        bm25_hit = {
            "_id": "w1",
            "_score": 5.0,
            "matched_queries": ["hit_summary"],
            "_source": {
                "asset_id": "w1",
                "modality": "text",
                "domain_label": "general",
                "summary": "충전 테스트",
            },
        }
        knn_hit = {"_id": "w1", "_score": 0.5, "_source": {"asset_id": "w1", "modality": "text"}}
        fused = fuse_hybrid([bm25_hit], [knn_hit], weights=(0.5, 0.5))
        self.assertEqual(len(fused), 1)
        self.assertTrue(fused[0].get("_bm25"))
        self.assertEqual(fused[0].get("matched_queries"), ["hit_summary"])


class EvidenceRescueIntegrationTest(unittest.TestCase):
    def _gate_fail_client(self, bm25_hits: list[dict]) -> _FakeMsearchClient:
        return _FakeMsearchClient(
            knn_by_label={"text": [_knn_hit_os("n1", 0.70)]},
            bm25_by_label={"text": bm25_hits},
        )

    def test_rescue_env_gate_default_off(self) -> None:
        hit = _bm25_hit_os("w1", 3.0)
        hit["matched_queries"] = ["hit_summary"]
        client = self._gate_fail_client([hit])
        buckets, _ = search_assets_os(
            client,
            "테스트",
            modalities=("text",),
            index="assets",
            embed_fn=lambda q, channel: [0.1] * 8,
            cutoff_enabled=True,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
            result_floor=0.99,
            evidence_rescue_enabled=False,
        )
        self.assertEqual([r["id"] for r in buckets["text"]], ["w1"])

    def test_restricted_weak_only_dropped_when_enabled(self) -> None:
        hit = _bm25_hit_os("w1", 3.0)
        hit["matched_queries"] = ["hit_summary", "hit_cross_meta"]
        client = self._gate_fail_client([hit])
        buckets, _ = search_assets_os(
            client,
            "테스트",
            modalities=("text",),
            index="assets",
            embed_fn=lambda q, channel: [0.1] * 8,
            cutoff_enabled=True,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
            result_floor=0.99,
            evidence_rescue_enabled=True,
        )
        self.assertEqual(buckets["text"], [])

    def test_restricted_strong_keep_when_enabled(self) -> None:
        hit = _bm25_hit_os("k1", 3.0)
        hit["matched_queries"] = ["hit_keywords"]
        client = self._gate_fail_client([hit])
        buckets, _ = search_assets_os(
            client,
            "테스트",
            modalities=("text",),
            index="assets",
            embed_fn=lambda q, channel: [0.1] * 8,
            cutoff_enabled=True,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
            result_floor=0.99,
            evidence_rescue_enabled=True,
        )
        self.assertEqual([r["id"] for r in buckets["text"]], ["k1"])

    def test_evidence_debug_meta_on_hit(self) -> None:
        hit = _bm25_hit_os("d1", 3.0)
        hit["matched_queries"] = ["hit_summary"]
        client = self._gate_fail_client([hit])
        buckets, _ = search_assets_os(
            client,
            "테스트",
            modalities=("text",),
            index="assets",
            embed_fn=lambda q, channel: [0.1] * 8,
            cutoff_enabled=True,
            cutoff_eps=0.9,
            cutoff_floor=0.9,
            result_floor=0.99,
            evidence_rescue_enabled=False,
            evidence_debug=True,
        )
        row = buckets["text"][0]
        self.assertEqual(row["matched_queries"], ["hit_summary"])
        self.assertIn("evidence_score", row)
        self.assertIn("keep_reason", row)
        self.assertFalse(row["gate_passed"])


if __name__ == "__main__":
    unittest.main()
