"""051 FR-401: 관계 골든 비회귀 게이트(RUN_RELATION_GOLDEN=1 에서만).

골든+동결 snapshot 으로 measure 를 재실행해 baseline 대비 candidate_recall·관계 recall·
isolation_accuracy 가 -EPS 이상임을 검증한다. LLM 0(snapshot 동결)·결정적(헌법 2·3조).
미설정(기본) 시 skip — 회귀 suite 0 영향(SC-004). 골든·baseline 은 로컬(gitignore·C4).
"""
import json
import os
import unittest
from pathlib import Path

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "relations"
_GOLDEN = _GOLDEN_DIR / "relation_golden.json"
_SNAPSHOT = _GOLDEN_DIR / "relation_snapshot.json"
_BASELINE = _GOLDEN_DIR / "baseline_report.json"
_EPS = 0.02


@unittest.skipUnless(os.environ.get("RUN_RELATION_GOLDEN") == "1",
                     "RUN_RELATION_GOLDEN=1 에서만(로컬 골든·baseline 필요)")
class RelationGoldenRegressionTest(unittest.TestCase):
    def setUp(self):
        for p in (_GOLDEN, _SNAPSHOT, _BASELINE):
            if not p.exists():
                self.skipTest(f"로컬 산출물 없음: {p.name}")
        from scripts.measure_relation_quality import _load_golden
        self.baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
        self.golden = _load_golden(str(_GOLDEN))

    def _measure(self) -> dict:
        # baseline 이 동결될 때 쓴 confidence_min(프로덕션 자동승인 임계)을 재사용해야
        # accepted 판정이 일치한다(0.0 으로 재면 isolation 이 항상 0 이라 baseline 과 어긋남).
        from scripts.measure_relation_quality import cmd_measure
        cmin = float(self.baseline.get("confidence_min", 0.0))
        return cmd_measure(self.golden, str(_SNAPSHOT), confidence_min=cmin)

    def test_candidate_recall_non_regression(self):
        cur = self._measure()["candidate_recall"]
        base = self.baseline["candidate_recall"]
        self.assertGreaterEqual(cur, base - _EPS, f"candidate_recall {cur} < {base}-{_EPS}")

    def test_relation_recall_and_isolation_non_regression(self):
        rm = self._measure()["relation_metrics"]
        b = self.baseline["relation_metrics"]
        self.assertGreaterEqual(rm["recall"], b["recall"] - _EPS,
                                f"relation recall {rm['recall']} < {b['recall']}-{_EPS}")
        self.assertGreaterEqual(rm["isolation_accuracy"], b["isolation_accuracy"] - _EPS,
                                f"isolation_accuracy {rm['isolation_accuracy']} < {b['isolation_accuracy']}-{_EPS}")


if __name__ == "__main__":
    unittest.main()
