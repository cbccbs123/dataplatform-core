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


class ChooseEncodingTruncationTest(unittest.TestCase):
    """2026-07-15 B5 — 64KiB 경계가 멀티바이트 문자 중간에 걸려도 utf-8 을 오판하지 않는다."""

    def test_utf8_cut_mid_char_still_utf8(self) -> None:
        # '가'(3바이트) 반복으로 65,536 바이트 지점이 문자 중간(65536 % 3 == 1)에 걸리는 파일.
        # 종전(일반 decode)엔 꼬리 실패 → cp949 로 오판. incremental(final=False)은 utf-8 유지.
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write("가".encode() * 30000)  # 90,000 바이트 > 64KiB
            p = Path(f.name)
        try:
            self.assertEqual(dl._choose_encoding(p, "utf-8"), "utf-8")
        finally:
            p.unlink()

    def test_truly_cp949_still_cp949(self) -> None:
        # 진짜 cp949 문서는 여전히 cp949 로 판정(회귀 보존) — 잘림 없는 소형 표본.
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write("한글 인코딩 확인".encode("cp949"))
            p = Path(f.name)
        try:
            self.assertEqual(dl._choose_encoding(p, "utf-8"), "cp949")
        finally:
            p.unlink()


if __name__ == "__main__":
    unittest.main()
