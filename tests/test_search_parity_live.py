"""045 Phase A — search_assets_os·msearch 경로 동작 동일 + 라이브 OS parity."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from tests.test_opensearch_search import SearchAssetsOsMsearchTest

_BASELINE = Path(__file__).resolve().parent / "fixtures/search/parity_snapshots/pre_refactor_baseline.json"


class SearchAssetsOsBucketPolicyParityTest(unittest.TestCase):
    """기존 SearchAssetsOsMsearchTest 시나리오가 리팩터 후에도 동일한지 재검증."""
    def _fresh_base(self) -> SearchAssetsOsMsearchTest:
        t = SearchAssetsOsMsearchTest()
        t.setUp()
        return t

    def test_embed_and_msearch_unchanged(self) -> None:
        self._fresh_base().test_embeds_query_once()
        self._fresh_base().test_single_msearch_call()
        self._fresh_base().test_gate_pass_keeps_bucket()
        self._fresh_base().test_gate_fail_empties_bucket_but_records_meta()
        self._fresh_base().test_per_result_cut_keeps_bm25_or_high_cos()
        self._fresh_base().test_lexical_rescue_keeps_only_bm25_rows_when_gate_fails()
        self._fresh_base().test_no_lexical_no_gate_means_empty()
        self._fresh_base().test_cutoff_disabled_no_gate_no_cut()

    def test_gate_meta_fields_stable(self) -> None:
        base = self._fresh_base()
        buckets, meta = base._run()
        for label in ("text", "audio"):
            gm = meta[label]
            self.assertEqual(
                set(gm),
                {"top", "baseline", "gate_passed", "lexical_evidence", "cut_count", "error"},
            )
            self.assertIsInstance(buckets[label], list)


class LiveSearchParityTest(unittest.TestCase):
    """실 OS·run_search 경로 — 리팩터 전 스냅샷과 meta·결과 id 동일."""

    @classmethod
    def setUpClass(cls) -> None:
        if os.getenv("RUN_SEARCH_PARITY") != "1":
            raise unittest.SkipTest("RUN_SEARCH_PARITY=1 일 때만 라이브 parity 실행")
        if not _BASELINE.is_file():
            raise unittest.SkipTest(f"baseline 없음: {_BASELINE}")

    def _live_slim(self, query: str) -> dict:
        raw = subprocess.check_output(
            [sys.executable, "-m", "src.app.run_search", "--env", "dev", "--query", query, "--limit", "5"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        d = json.loads(raw)
        return {
            "meta": d.get("meta"),
            "results": {
                bucket: [{"id": r.get("id"), "similarity": r.get("similarity")} for r in rows[:5]]
                for bucket, rows in (d.get("results") or {}).items()
            },
        }

    def _slim_from_baseline_entry(self, entry: dict) -> dict:
        return {
            "meta": entry.get("meta"),
            "results": {
                bucket: [{"id": r["id"], "similarity": r["similarity"]} for r in rows]
                for bucket, rows in (entry.get("results") or {}).items()
            },
        }

    def test_live_matches_pre_refactor_baseline(self) -> None:
        baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
        for query, expected_raw in baseline.items():
            with self.subTest(query=query):
                expected = self._slim_from_baseline_entry(expected_raw)
                live = self._live_slim(query)
                self.assertEqual(live["meta"], expected["meta"])
                self.assertEqual(live["results"], expected["results"])


if __name__ == "__main__":
    unittest.main()
