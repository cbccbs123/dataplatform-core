"""069 T004(P1-8) — data_loader._choose_encoding 부분 읽기(64KiB) 검증. 실대용량 무부담."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.file import data_loader as dl


class TestChooseEncodingPartialRead(unittest.TestCase):
    def test_reads_partially_not_whole_file(self) -> None:
        # 큰 파일(~3MiB)에서도 read(65536) 부분 읽기 — read_bytes(전체 로드) 경로 미사용.
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write("가나다라".encode() * 262144)
            p = Path(f.name)
        try:
            with patch.object(Path, "read_bytes") as m_rb:
                enc = dl._choose_encoding(p, "utf-8")
            m_rb.assert_not_called()  # P1-8: 전체 로드 제거 확인
            self.assertEqual(enc, "utf-8")
        finally:
            p.unlink()

    def test_cp949_detection_unchanged(self) -> None:
        # 판정 동작 동일성: utf-8 실패 → cp949 성공 경로 보존(기존 계약 불변).
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write("한글 인코딩 테스트".encode("cp949"))
            p = Path(f.name)
        try:
            self.assertEqual(dl._choose_encoding(p, "utf-8"), "cp949")
        finally:
            p.unlink()


if __name__ == "__main__":
    unittest.main()
