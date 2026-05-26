"""F-5.1 의료 분류 cascade 단위 테스트.

stage1(시그니처)·stage2(어휘) 는 결정적이라 실데이터 픽스처로, stage3(온프레미스 LLM)·cascade 단축은
mock 으로 검증한다. 외부 모델/네트워크 불필요.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _fake_cfg():
    return SimpleNamespace(openai_base_url="http://onprem/v1", openai_api_key="EMPTY", meta_model="gemma")


def _fake_openai_client(content: str):
    """chat.completions.create 가 content 를 반환하는 mock OpenAI 클라이언트."""
    choice = mock.MagicMock()
    choice.message.content = content
    resp = mock.MagicMock()
    resp.choices = [choice]
    client = mock.MagicMock()
    client.chat.completions.create.return_value = resp
    return client

from src.classify import cascade, stage1_magic, stage2_zeroshot, stage3_gemma
from src.classify.types import DOMAIN_GENERAL, DOMAIN_MEDICAL, DOMAIN_REVIEW


class TmpBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, data: bytes) -> str:
        p = self.dir / name
        p.write_bytes(data)
        return str(p)


class TestStage1(TmpBase):
    def test_dicom_signature(self) -> None:
        path = self._write("a.dcm", b"\x00" * 128 + b"DICM" + b"\x00" * 100)
        label, sig = stage1_magic.detect(path)
        self.assertEqual(label, DOMAIN_MEDICAL)
        self.assertEqual(sig["signature"], "dicom")

    def test_hl7v2_signature(self) -> None:
        path = self._write("a.hl7", b"MSH|^~\\&|HIS|HOSP|...\r")
        label, sig = stage1_magic.detect(path)
        self.assertEqual(label, DOMAIN_MEDICAL)
        self.assertEqual(sig["signature"], "hl7v2")

    def test_fhir_signature(self) -> None:
        path = self._write("a.json", b'{"resourceType": "Patient", "id": "123"}')
        label, sig = stage1_magic.detect(path)
        self.assertEqual(label, DOMAIN_MEDICAL)
        self.assertEqual(sig["resourceType"], "Patient")

    def test_non_medical_none(self) -> None:
        path = self._write("a.txt", "오늘 날씨가 좋네요 hello world".encode("utf-8"))
        label, _ = stage1_magic.detect(path)
        self.assertIsNone(label)

    def test_fhir_unknown_resourcetype_none(self) -> None:
        path = self._write("a.json", b'{"resourceType": "Widget"}')
        label, _ = stage1_magic.detect(path)
        self.assertIsNone(label)


class TestStage2(unittest.TestCase):
    def test_two_or_more_hits_medical(self) -> None:
        label, sig = stage2_zeroshot.score("환자 진단 결과 보고서")
        self.assertEqual(label, DOMAIN_MEDICAL)
        self.assertGreaterEqual(sig["medical_term_hits"], 2)

    def test_zero_hits_general(self) -> None:
        label, sig = stage2_zeroshot.score("주식 시장 동향 분석")
        self.assertEqual(label, DOMAIN_GENERAL)
        self.assertEqual(sig["medical_term_hits"], 0)

    def test_one_hit_ambiguous(self) -> None:
        label, sig = stage2_zeroshot.score("환자 대기실 안내")
        self.assertIsNone(label)
        self.assertEqual(sig["medical_term_hits"], 1)

    def test_english_terms_case_insensitive(self) -> None:
        label, _ = stage2_zeroshot.score("Patient DIAGNOSIS summary")
        self.assertEqual(label, DOMAIN_MEDICAL)


class TestStage3(unittest.TestCase):
    def test_onprem_llm_medical(self) -> None:
        with mock.patch("src.config.settings.get_current_settings", return_value=_fake_cfg()), \
                mock.patch("openai.OpenAI", return_value=_fake_openai_client('{"label": "medical"}')):
            label, sig = stage3_gemma.classify("환자 진단")
        self.assertEqual(label, DOMAIN_MEDICAL)
        self.assertEqual(sig["stage3"], "llm")

    def test_onprem_unclear_returns_review(self) -> None:
        with mock.patch("src.config.settings.get_current_settings", return_value=_fake_cfg()), \
                mock.patch("openai.OpenAI", return_value=_fake_openai_client('{"label": "blah"}')):
            label, sig = stage3_gemma.classify("애매한 텍스트")
        self.assertEqual(label, DOMAIN_REVIEW)
        self.assertEqual(sig["stage3"], "llm_unclear")

    def test_error_returns_review(self) -> None:
        with mock.patch("src.config.settings.get_current_settings", side_effect=RuntimeError("not init")):
            label, sig = stage3_gemma.classify("환자")
        self.assertEqual(label, DOMAIN_REVIEW)
        self.assertEqual(sig["stage3"], "error")


class TestCascade(TmpBase):
    def test_stage1_short_circuit_medical(self) -> None:
        path = self._write("a.dcm", b"\x00" * 128 + b"DICM")
        r = cascade.classify(path, "unknown")
        self.assertEqual(r.final_label, DOMAIN_MEDICAL)
        self.assertEqual(r.decided_stage, 1)
        self.assertEqual(r.confidence, 1.0)

    def test_stage2_medical_from_text_content(self) -> None:
        path = self._write("note.txt", "환자 진단 처방 내역".encode("utf-8"))
        r = cascade.classify(path, "txt")
        self.assertEqual(r.final_label, DOMAIN_MEDICAL)
        self.assertEqual(r.decided_stage, 2)

    def test_stage2_general(self) -> None:
        path = self._write("note.txt", "여행 후기와 맛집 추천".encode("utf-8"))
        r = cascade.classify(path, "txt")
        self.assertEqual(r.final_label, DOMAIN_GENERAL)
        self.assertEqual(r.decided_stage, 2)

    def test_ambiguous_goes_to_stage3(self) -> None:
        # stage1 None, stage2 모호(None) → stage3 호출.
        with mock.patch.object(cascade.stage1_magic, "detect", return_value=(None, {})), \
                mock.patch.object(cascade.stage2_zeroshot, "score", return_value=(None, {"medical_term_hits": 1})), \
                mock.patch.object(cascade.stage3_gemma, "classify", return_value=(DOMAIN_GENERAL, {"stage3": "llm"})) as s3:
            r = cascade.classify("/x/y.bin", "image")
        s3.assert_called_once()
        self.assertEqual(r.decided_stage, 3)
        self.assertEqual(r.final_label, DOMAIN_GENERAL)


if __name__ == "__main__":
    unittest.main()
