"""도메인-불가지 cascade 엔진 — provider 주입으로 다도메인 검증(DB/네트워크 불필요)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.classify import cascade
from src.classify.profiles import DomainProfile, SigHit
from src.classify.types import DOMAIN_GENERAL, DOMAIN_MEDICAL, DOMAIN_REVIEW


def _dicom(head: bytes):
    return SigHit("dicom") if len(head) >= 132 and head[128:132] == b"DICM" else None


class _Provider:
    def __init__(self, profiles):
        self._p = profiles
    def all_profiles(self):
        return list(self._p)


MEDICAL = DomainProfile("medical", frozenset({"환자", "진단", "처방"}), "medical", (_dicom,))
LEGAL = DomainProfile("legal", frozenset({"계약", "소송", "판결"}), "legal", ())


class TestEngine(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.provider = _Provider([MEDICAL, LEGAL])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, data: bytes) -> str:
        p = self.dir / name
        p.write_bytes(data)
        return str(p)

    def test_stage1_signature_unique_match(self) -> None:
        path = self._write("a.dcm", b"\x00" * 128 + b"DICM")
        r = cascade.classify(path, "unknown", provider=self.provider)
        self.assertEqual(r.final_label, DOMAIN_MEDICAL)
        self.assertEqual(r.decided_stage, 1)
        self.assertEqual(r.confidence, 1.0)

    def test_stage2_top_domain_confirmed(self) -> None:
        path = self._write("n.txt", "환자 진단 처방 내역".encode("utf-8"))
        r = cascade.classify(path, "txt", provider=self.provider)
        self.assertEqual(r.final_label, DOMAIN_MEDICAL)
        self.assertEqual(r.decided_stage, 2)
        self.assertEqual(r.stage2_scores["medical"]["hits"], 3)

    def test_stage2_other_domain_confirmed(self) -> None:
        path = self._write("n.txt", "계약 소송 판결 요지".encode("utf-8"))
        r = cascade.classify(path, "txt", provider=self.provider)
        self.assertEqual(r.final_label, "legal")
        self.assertEqual(r.decided_stage, 2)

    def test_stage2_zero_hits_general(self) -> None:
        path = self._write("n.txt", "여행 후기와 맛집 추천".encode("utf-8"))
        r = cascade.classify(path, "txt", provider=self.provider)
        self.assertEqual(r.final_label, DOMAIN_GENERAL)
        self.assertEqual(r.decided_stage, 2)

    def test_stage3_called_when_ambiguous(self) -> None:
        path = self._write("n.txt", "환자 대기실 안내".encode("utf-8"))
        calls = {}
        def fake_s3(text, labels, **kw):
            calls["labels"] = labels
            return DOMAIN_GENERAL, {"stage3": "llm"}
        r = cascade.classify(path, "txt", provider=self.provider, _llm_classify=fake_s3)
        self.assertEqual(r.decided_stage, 3)
        self.assertEqual(r.final_label, DOMAIN_GENERAL)
        self.assertIn("medical", calls["labels"])
        self.assertIn("legal", calls["labels"])
        self.assertIn(DOMAIN_GENERAL, calls["labels"])

    def test_stage1_conflict_falls_through_to_stage2(self) -> None:
        med2 = DomainProfile("medical", MEDICAL.lexicon, "medical", (_dicom,))
        also = DomainProfile("imaging", frozenset(), "imaging", (_dicom,))
        prov = _Provider([med2, also])
        path = self._write("a.dcm", b"\x00" * 128 + b"DICM")
        r = cascade.classify(path, "unknown", provider=prov)
        self.assertNotEqual(r.decided_stage, 1)

    def test_stage3_unclear_returns_review(self) -> None:
        path = self._write("n.txt", "환자 대기실 안내".encode("utf-8"))
        r = cascade.classify(
            path, "txt", provider=self.provider,
            _llm_classify=lambda t, l, **k: (DOMAIN_REVIEW, {"stage3": "llm_unclear"}),
        )
        self.assertEqual(r.final_label, DOMAIN_REVIEW)


if __name__ == "__main__":
    unittest.main()
