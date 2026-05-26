"""F-5.1 분류용 텍스트 추출(text_probe) 단위 테스트.

문서(평문)는 실제 읽기로, 이미지 OCR·라우팅은 mock 으로 검증(OCR 정확도는 e2e 에서).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.classify import text_probe


class TestTextProbe(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_plain_txt_read(self) -> None:
        p = self.dir / "a.txt"
        p.write_text("환자 진단 처방", encoding="utf-8")
        self.assertIn("진단", text_probe.extract_text_for_classification(str(p), "txt"))

    def test_json_read(self) -> None:
        p = self.dir / "a.json"
        p.write_text('{"diagnosis": "fever"}', encoding="utf-8")
        self.assertIn("diagnosis", text_probe.extract_text_for_classification(str(p), "json"))

    def test_image_uses_ocr(self) -> None:
        # 이미지 모달리티는 OCR 경로. pytesseract 를 mock 해 배선만 검증.
        p = self.dir / "x.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")  # 더미(실제 OCR은 mock)
        with mock.patch("pytesseract.image_to_string", return_value="처방전 약품 환자"), \
                mock.patch("PIL.Image.open"):
            out = text_probe.extract_text_for_classification(str(p), "image")
        self.assertEqual(out, "처방전 약품 환자")

    def test_audio_returns_empty(self) -> None:
        self.assertEqual(text_probe.extract_text_for_classification("/x/a.mp3", "audio"), "")

    def test_ocr_failure_returns_empty(self) -> None:
        with mock.patch("pytesseract.image_to_string", side_effect=RuntimeError("ocr fail")):
            self.assertEqual(text_probe.extract_text_for_classification("/x/a.png", "image"), "")


if __name__ == "__main__":
    unittest.main()
