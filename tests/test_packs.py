"""v2 단계 B — 도메인 팩 단위 테스트."""
from __future__ import annotations

import unittest

from src.pipeline.packs import GENERAL_PACK, MEDICAL_PACK, for_domain


class TestDomainPacks(unittest.TestCase):
    def test_general_and_medical_fields(self) -> None:
        self.assertEqual(GENERAL_PACK.policy, "general_default")
        self.assertEqual(MEDICAL_PACK.policy, "medical_strict")
        for pack in (GENERAL_PACK, MEDICAL_PACK):
            self.assertEqual(set(pack.per_asset), {"classify", "extract", "embed", "persist"})

    def test_for_domain_mapping_and_fallback(self) -> None:
        self.assertIs(for_domain("general"), GENERAL_PACK)
        self.assertIs(for_domain("medical"), MEDICAL_PACK)
        self.assertIs(for_domain("review"), GENERAL_PACK)   # 미지정/유보는 일반으로 보수적 폴백
        self.assertIs(for_domain("unknown"), GENERAL_PACK)


if __name__ == "__main__":
    unittest.main()
