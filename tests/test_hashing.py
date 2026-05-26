"""파일 해시(dedup) 단위 테스트."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.file.hashing import file_hash_and_size, sha256_file


class TestHashing(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.f = Path(self._tmp.name) / "x.bin"
        self.content = b"hello dedup \x00\x01\x02" * 1000
        self.f.write_bytes(self.content)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sha256_matches_hashlib(self) -> None:
        self.assertEqual(sha256_file(str(self.f)), hashlib.sha256(self.content).hexdigest())

    def test_hash_and_size(self) -> None:
        h, size = file_hash_and_size(str(self.f))
        self.assertEqual(len(h), 64)
        self.assertEqual(size, len(self.content))

    def test_same_content_same_hash(self) -> None:
        other = Path(self._tmp.name) / "y.bin"
        other.write_bytes(self.content)
        self.assertEqual(sha256_file(str(self.f)), sha256_file(str(other)))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            sha256_file(str(Path(self._tmp.name) / "nope.bin"))


if __name__ == "__main__":
    unittest.main()
