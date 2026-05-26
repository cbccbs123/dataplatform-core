"""UUIDv7 생성기 단위 테스트."""

from __future__ import annotations

import unittest
import uuid

from src.database.ids import uuid7


class TestUuid7(unittest.TestCase):
    def test_version_and_variant(self) -> None:
        u = uuid7()
        self.assertEqual(u.version, 7)
        self.assertEqual((u.int >> 62) & 0b11, 0b10)  # RFC 4122 variant

    def test_uniqueness(self) -> None:
        self.assertEqual(len({uuid7() for _ in range(1000)}), 1000)

    def test_time_ordered(self) -> None:
        # 생성 순서대로 대체로 증가(타임스탬프 prefix). 연속 생성은 비감소.
        a = uuid7()
        b = uuid7()
        self.assertLessEqual(a.bytes[:6], b.bytes[:6])

    def test_is_uuid_instance(self) -> None:
        self.assertIsInstance(uuid7(), uuid.UUID)


if __name__ == "__main__":
    unittest.main()
