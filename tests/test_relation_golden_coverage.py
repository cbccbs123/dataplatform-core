import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.measure_relation_quality import _dump_report

from src.relations.quality.metrics import isolated_candidates


class IsolatedCandidatesTest(unittest.TestCase):
    def test_returns_registered_minus_candidates_sorted(self):
        # registered 5건 중 후보에 등장한 3건(a1·a2·a3) 제외 → 고립 = a4·a5(정렬).
        reg = {"a3", "a1", "a5", "a2", "a4"}
        cand = {"a1", "a2", "a3"}
        self.assertEqual(isolated_candidates(reg, cand), ["a4", "a5"])

    def test_no_isolated_when_all_have_candidates(self):
        self.assertEqual(isolated_candidates({"a1", "a2"}, {"a1", "a2"}), [])

    def test_empty_registered(self):
        self.assertEqual(isolated_candidates(set(), {"a1"}), [])

    def test_candidate_ids_outside_registered_ignored(self):
        # 후보에만 있고 registered 가 아닌 id(b9)는 결과에 영향 없음.
        self.assertEqual(isolated_candidates({"a1", "a2"}, {"a1", "b9"}), ["a2"])


class DumpReportTest(unittest.TestCase):
    def test_deterministic_sorted_bytes(self):
        report = {"relation_metrics": {"recall": 0.83, "isolation_accuracy": 1.0},
                  "candidate_recall": 0.83, "config": {"min_sim": 0.2}}
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = os.path.join(d, "a.json"), os.path.join(d, "b.json")
            _dump_report(report, p1)
            _dump_report(report, p2)
            b1 = Path(p1).read_text(encoding="utf-8")
            b2 = Path(p2).read_text(encoding="utf-8")
        self.assertEqual(b1, b2)                       # 같은 입력 → byte 동일(결정적)
        self.assertEqual(json.loads(b1)["relation_metrics"]["recall"], 0.83)
        # sort_keys: 최상위 키가 정렬돼 있어야(candidate_recall < config < relation_metrics)
        self.assertLess(b1.index('"candidate_recall"'), b1.index('"relation_metrics"'))


if __name__ == "__main__":
    unittest.main()
