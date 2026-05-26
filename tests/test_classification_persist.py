"""F-5.1 분류 결과 영속화 단위 테스트(mock Connection)."""

from __future__ import annotations

import unittest
import uuid
from unittest import mock

from src.classify.types import ClassificationResult
from src.registry.classification_persist import record_classification

_AID = uuid.UUID("018f0000-0000-7000-8000-000000000001")


def _conn():
    conn = mock.MagicMock()
    return conn, conn.cursor.return_value.__enter__.return_value


def _execs(cur):
    return [c.args[0] for c in cur.execute.call_args_list]


class TestRecordClassification(unittest.TestCase):
    def test_medical_inserts_and_updates_domain(self) -> None:
        conn, cur = _conn()
        r = ClassificationResult(final_label="medical", confidence=1.0, decided_stage=1, stage1_scores={"signature": "dicom"})
        record_classification(conn, _AID, r)
        sqls = _execs(cur)
        self.assertTrue(any("INSERT INTO asset_classification" in s for s in sqls))
        self.assertTrue(any("UPDATE asset SET domain_label" in s for s in sqls))
        # medical 은 unresolved_pool 미적재
        self.assertFalse(any("unresolved_pool" in s for s in sqls))

    def test_review_inserts_unresolved_pool(self) -> None:
        conn, cur = _conn()
        r = ClassificationResult(final_label="review", confidence=0.0, decided_stage=3)
        record_classification(conn, _AID, r)
        sqls = _execs(cur)
        self.assertTrue(any("INSERT INTO unresolved_pool" in s for s in sqls))

    def test_general_no_unresolved_pool(self) -> None:
        conn, cur = _conn()
        r = ClassificationResult(final_label="general", confidence=0.7, decided_stage=2)
        record_classification(conn, _AID, r)
        self.assertFalse(any("unresolved_pool" in s for s in _execs(cur)))


if __name__ == "__main__":
    unittest.main()
