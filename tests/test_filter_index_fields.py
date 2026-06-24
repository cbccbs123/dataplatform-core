"""045 Phase B — filter_kw·filter_date 파생 필드 단위."""

from __future__ import annotations

import unittest
from datetime import date, datetime

from src.search.filter_index_fields import (
    build_filter_index_fields,
    derive_file_ext,
    derive_source_dataset,
)


class DeriveFileExtTest(unittest.TestCase):
    def test_last_extension_lowercase(self) -> None:
        self.assertEqual(derive_file_ext("/data/a.stt.txt"), "txt")

    def test_no_extension(self) -> None:
        self.assertIsNone(derive_file_ext("/data/noext"))
        self.assertIsNone(derive_file_ext("/data/trailing."))


class DeriveSourceDatasetTest(unittest.TestCase):
    def test_sample_data_buckets(self) -> None:
        self.assertEqual(derive_source_dataset("/x/sample_data/data2/foo.mp4"), "data2")

    def test_wikipedia_youtube(self) -> None:
        self.assertEqual(derive_source_dataset("/corpus/wikipedia/article.txt"), "wikipedia")
        self.assertEqual(derive_source_dataset("/yt/youtube/abc.mp4"), "youtube")

    def test_unknown(self) -> None:
        self.assertEqual(derive_source_dataset("/random/path/file.pdf"), "unknown")


class BuildFilterIndexFieldsTest(unittest.TestCase):
    def test_full_shape(self) -> None:
        out = build_filter_index_fields(
            fs_path="/sample_data/data1/doc.pdf",
            created_at=datetime(2026, 3, 15, 12, 0, 0),
        )
        self.assertEqual(
            out,
            {
                "filter_kw": {"file_ext": "pdf", "source_dataset": "data1"},
                "filter_date": {"created_at": "2026-03-15"},
            },
        )

    def test_date_only_created_at(self) -> None:
        out = build_filter_index_fields(fs_path="/x/y.txt", created_at=date(2025, 12, 1))
        self.assertEqual(out["filter_date"], {"created_at": "2025-12-01"})


if __name__ == "__main__":
    unittest.main()
