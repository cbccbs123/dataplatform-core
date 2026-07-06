"""scripts/seed_topic_registry.py 순수 함수 단위 테스트 (spec 058 T501).

LLM/DB 불필요 — 파싱·검증·정규화·초안 구성·요약의 결정적 순수 함수만 검증한다.
"""
from __future__ import annotations

import unittest

from scripts.seed_topic_registry import (
    build_clustering_prompt,
    build_draft,
    build_topic_input,
    parse_groups,
    summarize_lines,
    validate_and_normalize_groups,
)


class TestBuildTopicInput(unittest.TestCase):
    def test_sorted_by_freq_desc_then_ko_asc(self):
        freq = {"음식": 77, "요리": 243, "식품": 33, "천문": 5, "천문학": 5}
        en = {"요리": {"cooking": 200, "cuisine": 43}, "음식": {"food": 77}}
        out = build_topic_input(freq, en)
        # 빈도 desc → 라벨 asc(동빈도 천문/천문학은 라벨 asc)
        self.assertEqual([o["topic_ko"] for o in out], ["요리", "음식", "식품", "천문", "천문학"])
        # en 변형은 빈도 desc → 알파벳 asc
        self.assertEqual(out[0]["en_variants"], ["cooking", "cuisine"])
        self.assertEqual(out[2]["en_variants"], [])

    def test_prompt_lists_every_topic(self):
        freq = {"요리": 243, "음식": 77}
        prompt = build_clustering_prompt(build_topic_input(freq, {}))
        self.assertIn("요리 (freq=243)", prompt)
        self.assertIn("음식 (freq=77)", prompt)
        self.assertIn("groups", prompt)


class TestParseGroups(unittest.TestCase):
    def test_parses_wellformed(self):
        raw = {
            "groups": [
                {"canonical_ko": "요리", "canonical_en": "cooking", "members": ["요리", "음식", "식품"]},
            ]
        }
        groups = parse_groups(raw)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["canonical_ko"], "요리")
        self.assertEqual(groups[0]["members"], ["요리", "음식", "식품"])

    def test_skips_malformed(self):
        raw = {
            "groups": [
                {"canonical_ko": "", "members": ["x"]},  # 빈 canonical → 스킵
                {"canonical_ko": "요리"},  # members 없음 → 스킵
                "not-a-dict",  # → 스킵
                {"canonical_ko": "천문", "canonical_en": "astronomy", "members": ["천문", "천문학"]},
            ]
        }
        groups = parse_groups(raw)
        self.assertEqual([g["canonical_ko"] for g in groups], ["천문"])

    def test_non_list_groups_returns_empty(self):
        self.assertEqual(parse_groups({}), [])
        self.assertEqual(parse_groups({"groups": "x"}), [])


class TestValidateAndNormalize(unittest.TestCase):
    def _freq(self):
        return {"요리": 243, "음식": 77, "식품": 33, "천문": 5, "천문학": 5, "여행": 20}

    def test_merge_group_total_freq_and_member_order(self):
        freq = self._freq()
        groups = [
            {"canonical_ko": "요리", "canonical_en": "cooking", "members": ["요리", "음식", "식품"]},
        ]
        norm, rep = validate_and_normalize_groups(groups, freq, {})
        cook = next(g for g in norm if g["canonical_ko"] == "요리")
        self.assertEqual(cook["total_freq"], 243 + 77 + 33)
        # member 는 빈도 desc → 라벨 asc
        self.assertEqual(cook["members"], ["요리", "음식", "식품"])

    def test_missing_topics_added_as_singletons_and_coverage_complete(self):
        freq = self._freq()
        # LLM 이 여행·천문·천문학을 누락
        groups = [{"canonical_ko": "요리", "canonical_en": "cooking", "members": ["요리", "음식", "식품"]}]
        norm, rep = validate_and_normalize_groups(groups, freq, {})
        canos = {g["canonical_ko"] for g in norm}
        self.assertIn("여행", canos)
        self.assertIn("천문", canos)
        self.assertTrue(rep["coverage_complete"])
        self.assertEqual(set(rep["missing_topics"]), {"여행", "천문", "천문학"})
        # 모든 입력이 정확히 1개 그룹에
        covered = [m for g in norm for m in g["members"]]
        self.assertEqual(sorted(covered), sorted(freq))
        self.assertEqual(len(covered), len(set(covered)))

    def test_duplicated_member_kept_in_first_group_only(self):
        freq = self._freq()
        groups = [
            {"canonical_ko": "요리", "canonical_en": "cooking", "members": ["요리", "음식"]},
            {"canonical_ko": "식품", "canonical_en": "food", "members": ["식품", "음식"]},  # 음식 중복
        ]
        norm, rep = validate_and_normalize_groups(groups, freq, {})
        self.assertIn("음식", rep["duplicated_members"])
        # 음식은 첫 그룹(요리)에만
        cook = next(g for g in norm if g["canonical_ko"] == "요리")
        food = next(g for g in norm if g["canonical_ko"] == "식품")
        self.assertIn("음식", cook["members"])
        self.assertNotIn("음식", food["members"])

    def test_member_not_in_input_dropped(self):
        freq = self._freq()
        groups = [{"canonical_ko": "요리", "canonical_en": "cooking", "members": ["요리", "존재안함"]}]
        norm, rep = validate_and_normalize_groups(groups, freq, {})
        self.assertIn("존재안함", rep["members_not_in_input"])
        cook = next(g for g in norm if g["canonical_ko"] == "요리")
        self.assertNotIn("존재안함", cook["members"])

    def test_canonical_not_in_input_reported(self):
        freq = self._freq()
        groups = [{"canonical_ko": "미식", "canonical_en": "gourmet", "members": ["요리", "음식"]}]
        norm, rep = validate_and_normalize_groups(groups, freq, {})
        self.assertIn("미식", rep["canonical_not_in_input"])

    def test_merge_groups_sorted_before_singletons(self):
        freq = self._freq()
        groups = [
            {"canonical_ko": "천문", "canonical_en": "astronomy", "members": ["천문", "천문학"]},
            {"canonical_ko": "요리", "canonical_en": "cooking", "members": ["요리", "음식", "식품"]},
        ]
        norm, _ = validate_and_normalize_groups(groups, freq, {})
        merge_canos = [g["canonical_ko"] for g in norm if len(g["members"]) > 1]
        singleton_canos = [g["canonical_ko"] for g in norm if len(g["members"]) == 1]
        # 병합그룹이 모두 앞에, total_freq desc(요리 353 > 천문 10)
        self.assertEqual(merge_canos, ["요리", "천문"])
        # 병합그룹 인덱스 < 단독그룹 인덱스
        idx = [g["canonical_ko"] for g in norm]
        self.assertLess(idx.index("요리"), idx.index("여행"))
        self.assertTrue(set(singleton_canos))

    def test_best_en_fallback_when_group_en_missing(self):
        freq = {"요리": 243, "음식": 77}
        en = {"요리": {"cooking": 200, "cuisine": 43}}
        groups = [{"canonical_ko": "요리", "canonical_en": "", "members": ["요리", "음식"]}]
        norm, _ = validate_and_normalize_groups(groups, freq, en)
        cook = next(g for g in norm if g["canonical_ko"] == "요리")
        self.assertEqual(cook["canonical_en"], "cooking")


class TestBuildDraftAndSummary(unittest.TestCase):
    def test_draft_structure(self):
        freq = {"요리": 243, "음식": 77, "여행": 20}
        groups = [{"canonical_ko": "요리", "canonical_en": "cooking", "members": ["요리", "음식"]}]
        norm, rep = validate_and_normalize_groups(groups, freq, {})
        draft = build_draft(norm, rep, n_topics=len(freq), n_edges=340)
        self.assertEqual(draft["generated_from"], {"n_topics": 3, "n_edges": 340})
        self.assertIn("groups", draft)
        self.assertIn("_validation", draft)
        # 첫 그룹은 병합그룹
        self.assertGreater(len(draft["groups"][0]["members"]), 1)

    def test_summary_lists_merge_group(self):
        freq = {"요리": 243, "음식": 77, "여행": 20}
        groups = [{"canonical_ko": "요리", "canonical_en": "cooking", "members": ["요리", "음식"]}]
        norm, rep = validate_and_normalize_groups(groups, freq, {})
        draft = build_draft(norm, rep, n_topics=len(freq), n_edges=340)
        text = "\n".join(summarize_lines(draft))
        self.assertIn("요리", text)
        self.assertIn("음식", text)
        self.assertIn("병합", text)


if __name__ == "__main__":
    unittest.main()
