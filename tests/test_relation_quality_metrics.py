"""관계 품질 메트릭 순수 단위테스트 (spec 031 T003·T004·T005).

LLM/DB 불요 — 후보 recall(대칭 인정)·관계 P/R·kind/고립 정확도·임계 스윕을
합성 입력으로 전수 검증한다(SC-003). 대칭 kind는 양방향 인정(헌법 3조).
"""
import unittest

from src.relations.quality.metrics import candidate_recall


class TestCandidateRecall(unittest.TestCase):
    def test_symmetric_either_direction(self):
        pairs = [("a", "b"), ("c", "d")]
        # a→b 직접, d→c 역방향 — 대칭이므로 둘 다 회수.
        cands = {"a": {"b"}, "b": set(), "c": set(), "d": {"c"}}
        self.assertEqual(candidate_recall(pairs, cands), 1.0)

    def test_missed_pair(self):
        pairs = [("a", "b")]
        cands = {"a": {"z"}, "b": {"y"}}
        self.assertEqual(candidate_recall(pairs, cands), 0.0)

    def test_empty_pairs(self):
        self.assertEqual(candidate_recall([], {}), 0.0)


if __name__ == "__main__":
    unittest.main()
