"""053: ``modality_of`` file_kind→canonical 순수 매핑 단위 테스트 (FR-101/102/501).

저장(asset.modality) 경계 전용 매핑. 추출은 file_kind 를 그대로 쓰므로 이 매핑은
create_asset 저장 경계에서만 적용된다(A안). LLM·DB·파일 불필요한 순수 단위.
"""

from __future__ import annotations

import unittest

from src.file.file_type_defs import (
    ALLOWED_TEXT_META_FILE_KINDS,
    CANONICAL_MODALITIES,
    modality_of,
)

# 현 CHECK 10종 입력 → 기대 canonical 매핑(plan 매핑표) + 미지원값 폴백.
_CASES = [
    ("txt", "text"),
    ("pdf", "text"),
    ("json", "text"),
    ("word", "text"),
    ("excel", "text"),
    ("powerpoint", "text"),
    ("image", "image"),
    ("video", "video"),
    ("audio", "audio"),
    ("unknown", "unknown"),
    ("xyz", "unknown"),   # 미지원·판별불가 → 격리표식
]


class TestModalityOf(unittest.TestCase):
    def test_maps_each_file_kind_to_canonical(self) -> None:
        for file_kind, expected in _CASES:
            with self.subTest(file_kind=file_kind):
                self.assertEqual(modality_of(file_kind), expected)

    def test_text_meta_file_kinds_all_map_to_text(self) -> None:
        # ALLOWED_TEXT_META_FILE_KINDS(txt,pdf,json,word,excel,powerpoint) 전부 text 로.
        for file_kind in ALLOWED_TEXT_META_FILE_KINDS:
            with self.subTest(file_kind=file_kind):
                self.assertEqual(modality_of(file_kind), "text")

    def test_canonical_modalities_constant(self) -> None:
        self.assertEqual(
            CANONICAL_MODALITIES, ("text", "image", "video", "audio", "unknown")
        )

    def test_return_value_always_canonical(self) -> None:
        # 전 입력(케이스 + 임의값)에 대해 반환값은 항상 CANONICAL_MODALITIES 원소.
        for file_kind, _ in _CASES:
            with self.subTest(file_kind=file_kind):
                self.assertIn(modality_of(file_kind), CANONICAL_MODALITIES)
        self.assertIn(modality_of(""), CANONICAL_MODALITIES)


if __name__ == "__main__":
    unittest.main()
