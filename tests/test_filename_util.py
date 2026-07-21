"""filename_util(코어) 단위 테스트 — basename 추출·asset_id 프리픽스 제거·표시명 산출.

077 레포 분리 배경: 표시 유틸(``strip_asset_id_prefix``·``display_file_name``)을 코어(config)로
승격하면서 그 검증 테스트(``test_display_file_name``)는 백엔드 레포로 이관됐다. 이 함수들은 이제
코어 자산이므로 **코어 레포에서 직접** 검증한다(순수·표준 라이브러리만·DB/OS 불필요).
"""
from __future__ import annotations

import unittest

from src.config.filename_util import (
    basename_of,
    display_file_name,
    strip_asset_id_prefix,
)

# 실제 UUIDv7 형태 asset_id(8-4-4-4-12 hex) — 프리픽스 매칭 검증용.
_AID = "018f0000-0000-7000-8000-000000000277"


class BasenameOfTest(unittest.TestCase):
    def test_plain_path(self):
        self.assertEqual(basename_of("/data/archive/report.pdf"), "report.pdf")

    def test_backslash_normalized(self):
        # 윈도우 경로 백슬래시도 마지막 세그먼트를 취한다.
        self.assertEqual(basename_of(r"C:\docs\a.txt"), "a.txt")

    def test_trailing_slash_stripped(self):
        self.assertEqual(basename_of("/data/dir/"), "dir")

    def test_query_and_fragment_removed(self):
        self.assertEqual(basename_of("https://h/v/clip.mp4?t=3#x"), "clip.mp4")

    def test_empty_returns_empty(self):
        self.assertEqual(basename_of(""), "")

    def test_query_only_falls_back_to_tail(self):
        # 쿼리만 남아 앞이 비면 마지막 세그먼트로 폴백(기존 3벌 공통 ``or tail``).
        self.assertEqual(basename_of("http://h/?q=1"), "?q=1")

    def test_asset_id_prefix_not_stripped(self):
        # basename_of 는 표시용 strip 을 하지 않는다(색인·원본 경로 보존).
        self.assertEqual(basename_of(f"/a/{_AID}__orig.txt"), f"{_AID}__orig.txt")


class StripAssetIdPrefixTest(unittest.TestCase):
    def test_prefix_removed(self):
        self.assertEqual(strip_asset_id_prefix(f"{_AID}__orig.txt"), "orig.txt")

    def test_no_prefix_unchanged(self):
        self.assertEqual(strip_asset_id_prefix("orig.txt"), "orig.txt")

    def test_double_underscore_in_name_preserved(self):
        # 맨 앞이 UUID 형태가 아니면 ``__`` 가 있어도 건드리지 않는다.
        self.assertEqual(strip_asset_id_prefix("my__file.txt"), "my__file.txt")

    def test_empty_and_none(self):
        self.assertEqual(strip_asset_id_prefix(""), "")
        self.assertEqual(strip_asset_id_prefix(None), "")


class DisplayFileNameTest(unittest.TestCase):
    def test_path_with_prefix_returns_original(self):
        self.assertEqual(display_file_name(f"/data/{_AID}__orig.txt"), "orig.txt")

    def test_plain_path(self):
        self.assertEqual(display_file_name("/data/orig.txt"), "orig.txt")

    def test_none_and_empty_return_empty(self):
        self.assertEqual(display_file_name(None), "")
        self.assertEqual(display_file_name(""), "")


if __name__ == "__main__":
    unittest.main()
