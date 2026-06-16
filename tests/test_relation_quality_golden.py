"""골든 관계셋 파싱·검증 순수 단위테스트 (spec 031 T001).

LLM/DB 불요 — `parse_golden`이 유효 dict는 `Golden`으로, 결함은 `ValueError`로
처리하는지 전수 검증한다(SC-003).
"""
import unittest

from src.relations.quality.golden import Golden, GoldenPair, parse_golden


class TestParseGolden(unittest.TestCase):
    def test_parses_valid(self):
        g = parse_golden({
            "version": 1, "key_type": "fs_path",
            "pairs": [{"a": "/x/강의_1부.mp4", "b": "/x/강의_2부.mp4",
                       "kind": "same_series", "note": "연작"}],
            "isolated": ["/x/혼자.txt"]})
        self.assertIsInstance(g, Golden)
        self.assertEqual(g.key_type, "fs_path")
        self.assertEqual(
            g.pairs,
            (GoldenPair("/x/강의_1부.mp4", "/x/강의_2부.mp4", "same_series", "연작"),))
        self.assertEqual(g.isolated, ("/x/혼자.txt",))

    def test_rejects_bad_key_type(self):
        with self.assertRaises(ValueError):
            parse_golden({"version": 1, "key_type": "uuid", "pairs": [], "isolated": []})

    def test_rejects_missing_kind(self):
        with self.assertRaises(ValueError):
            parse_golden({"version": 1, "key_type": "fs_path",
                          "pairs": [{"a": "x", "b": "y"}], "isolated": []})

    def test_rejects_self_pair(self):
        with self.assertRaises(ValueError):
            parse_golden({"version": 1, "key_type": "fs_path",
                          "pairs": [{"a": "x", "b": "x", "kind": "same_series"}],
                          "isolated": []})


if __name__ == "__main__":
    unittest.main()
