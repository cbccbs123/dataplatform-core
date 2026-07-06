"""scripts/seed_topic_registry.py 순수 함수 단위 테스트 (spec 058 T501 · replay 교체).

LLM/DB 불필요 — replay 결과의 역집계·정렬·통계·초안 구성·요약의 결정적 순수 함수만 검증한다.
(실제 replay 는 ``canonicalize_topic`` 을 실 DB/LLM 로 재생하는 임퓨어 경로이며 여기서 다루지 않는다.)
"""
from __future__ import annotations

import unittest

from scripts.seed_topic_registry import (
    build_draft,
    build_replay_groups,
    build_replay_order,
    build_stats,
    summarize_lines,
)


class TestBuildReplayOrder(unittest.TestCase):
    def test_freq_desc_then_ko_asc(self):
        # 처리 순서 = 런타임 수집 순서 근사(빈도 desc → topic_ko asc·결정적).
        freq = {"음식": 77, "요리": 243, "식품": 33, "천문": 5, "천문학": 5}
        order = build_replay_order(freq)
        self.assertEqual(order, ["요리", "음식", "식품", "천문", "천문학"])


class TestBuildReplayGroups(unittest.TestCase):
    def _resolutions_cooking(self):
        # 요리(첫 NEW) ← 음식·식품(judge 매칭), 여행(별도 NEW)
        return [
            {"raw_ko": "요리", "canonical_ko": "요리"},
            {"raw_ko": "음식", "canonical_ko": "요리"},
            {"raw_ko": "식품", "canonical_ko": "요리"},
            {"raw_ko": "여행", "canonical_ko": "여행"},
        ]

    def _freq(self):
        return {"요리": 243, "음식": 77, "식품": 33, "여행": 20}

    def test_reverse_aggregation_members_and_total_freq(self):
        groups = build_replay_groups(
            self._resolutions_cooking(),
            self._freq(),
            {},
            {"요리": "cooking", "여행": "travel"},
        )
        cook = next(g for g in groups if g["canonical_ko"] == "요리")
        # members = 그 정본으로 해소된 raw 전부(정본 포함)·빈도 desc → 라벨 asc
        self.assertEqual(cook["members"], ["요리", "음식", "식품"])
        self.assertEqual(cook["total_freq"], 243 + 77 + 33)
        self.assertEqual(cook["canonical_en"], "cooking")

    def test_decided_by_llm_for_merge_new_for_singleton(self):
        groups = build_replay_groups(
            self._resolutions_cooking(), self._freq(), {}, {}
        )
        cook = next(g for g in groups if g["canonical_ko"] == "요리")
        travel = next(g for g in groups if g["canonical_ko"] == "여행")
        self.assertEqual(cook["decided_by"], "llm")  # judge 병합
        self.assertEqual(travel["decided_by"], "new")  # 단독 신규

    def test_merge_groups_sorted_before_singletons_then_total_freq_desc(self):
        resolutions = [
            {"raw_ko": "요리", "canonical_ko": "요리"},
            {"raw_ko": "음식", "canonical_ko": "요리"},
            {"raw_ko": "식품", "canonical_ko": "요리"},
            {"raw_ko": "천문", "canonical_ko": "천문"},
            {"raw_ko": "천문학", "canonical_ko": "천문"},
            {"raw_ko": "여행", "canonical_ko": "여행"},
        ]
        freq = {"요리": 243, "음식": 77, "식품": 33, "천문": 5, "천문학": 5, "여행": 20}
        groups = build_replay_groups(resolutions, freq, {}, {})
        merge = [g["canonical_ko"] for g in groups if len(g["members"]) > 1]
        # 병합그룹 우선 · total_freq desc(요리 353 > 천문 10)
        self.assertEqual(merge, ["요리", "천문"])
        idx = [g["canonical_ko"] for g in groups]
        self.assertLess(idx.index("요리"), idx.index("여행"))
        self.assertLess(idx.index("천문"), idx.index("여행"))

    def test_canonical_en_falls_back_to_best_en_when_registry_missing(self):
        # registry_en 스냅샷에 없으면 관측 en_variants 대표(빈도 desc → 알파벳 asc)로 폴백
        resolutions = [
            {"raw_ko": "요리", "canonical_ko": "요리"},
            {"raw_ko": "음식", "canonical_ko": "요리"},
        ]
        freq = {"요리": 243, "음식": 77}
        en = {"요리": {"cooking": 200, "cuisine": 43}}
        groups = build_replay_groups(resolutions, freq, en, {})
        cook = next(g for g in groups if g["canonical_ko"] == "요리")
        self.assertEqual(cook["canonical_en"], "cooking")

    def test_every_raw_appears_in_exactly_one_group(self):
        resolutions = self._resolutions_cooking()
        groups = build_replay_groups(resolutions, self._freq(), {}, {})
        covered = [m for g in groups for m in g["members"]]
        self.assertEqual(sorted(covered), sorted(self._freq()))
        self.assertEqual(len(covered), len(set(covered)))


class TestBuildStatsAndDraft(unittest.TestCase):
    def _groups(self):
        resolutions = [
            {"raw_ko": "요리", "canonical_ko": "요리"},
            {"raw_ko": "음식", "canonical_ko": "요리"},
            {"raw_ko": "여행", "canonical_ko": "여행"},
        ]
        return build_replay_groups(resolutions, {"요리": 243, "음식": 77, "여행": 20}, {}, {})

    def test_stats_counts(self):
        stats = build_stats(self._groups(), llm_calls=2)
        self.assertEqual(stats["n_canonical"], 2)
        self.assertEqual(stats["n_merged_groups"], 1)
        self.assertEqual(stats["n_singleton"], 1)
        self.assertEqual(stats["llm_calls"], 2)

    def test_draft_structure(self):
        groups = self._groups()
        stats = build_stats(groups, llm_calls=2)
        draft = build_draft(groups, stats, n_topics=3, n_edges=340)
        self.assertEqual(draft["mode"], "replay")
        self.assertEqual(draft["generated_from"], {"n_topics": 3, "n_edges": 340})
        self.assertIn("groups", draft)
        self.assertIn("stats", draft)
        # 첫 그룹은 병합그룹(members>1)
        self.assertGreater(len(draft["groups"][0]["members"]), 1)


class TestSummary(unittest.TestCase):
    def test_summary_lists_merge_group_and_llm_calls(self):
        resolutions = [
            {"raw_ko": "요리", "canonical_ko": "요리"},
            {"raw_ko": "음식", "canonical_ko": "요리"},
            {"raw_ko": "여행", "canonical_ko": "여행"},
        ]
        groups = build_replay_groups(resolutions, {"요리": 243, "음식": 77, "여행": 20}, {}, {})
        stats = build_stats(groups, llm_calls=2)
        draft = build_draft(groups, stats, n_topics=3, n_edges=340)
        text = "\n".join(summarize_lines(draft))
        self.assertIn("요리", text)
        self.assertIn("음식", text)
        self.assertIn("병합", text)
        self.assertIn("LLM", text)
        self.assertIn("2", text)  # llm_calls


if __name__ == "__main__":
    unittest.main()
