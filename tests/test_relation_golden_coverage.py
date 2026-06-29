import unittest

from src.relations.quality.metrics import isolated_candidates


class IsolatedCandidatesTest(unittest.TestCase):
    def test_returns_registered_minus_candidates_sorted(self):
        # registered 5건 중 후보에 등장한 3건(a1·a2·a3) 제외 → 고립 = a4·a5(정렬).
        reg = {"a3", "a1", "a5", "a2", "a4"}
        cand = {"a1", "a2", "a3"}
        self.assertEqual(isolated_candidates(reg, cand), ["a4", "a5"])

    def test_no_isolated_when_all_have_candidates(self):
        self.assertEqual(isolated_candidates({"a1", "a2"}, {"a1", "a2"}), [])

    def test_empty_registered(self):
        self.assertEqual(isolated_candidates(set(), {"a1"}), [])

    def test_candidate_ids_outside_registered_ignored(self):
        # 후보에만 있고 registered 가 아닌 id(b9)는 결과에 영향 없음.
        self.assertEqual(isolated_candidates({"a1", "a2"}, {"a1", "b9"}), ["a2"])


if __name__ == "__main__":
    unittest.main()
