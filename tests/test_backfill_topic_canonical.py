"""백필 topic 정규화 스크립트 단위 테스트(spec 058 G6 · T601/T602).

순수 재작성 계산(주입 seam)·SC 판정(01/03/04/07)·DB 해소 결선의 **LLM 0 불변식**을 실 DB 없이 덮는다.
해소 seam(alias 룩업·canonicalize_subtopic)은 가짜로 주입해 조립·규칙만 검증한다(docs/테스트_가이드 §seam).
"""
from __future__ import annotations

import unittest
from unittest import mock

from scripts.backfill_topic_canonical import (
    build_plan,
    format_report_lines,
    make_db_resolvers,
    rewrite_topic_row,
    sc03_topic_subtopic_overlap,
    sc04_modality_subtopics,
    sc07_distinct_topics,
    summarize_plan,
)


def _resolve_topic_via(alias: dict[str, str], reg_en: dict[str, str]):
    """가짜 resolve_topic — alias 정확일치(미스=원본 유지)·registry en 폴백."""

    def _fn(topic_ko: str, topic_en: str):
        hit = alias.get(topic_ko)
        alias_miss = hit is None and bool(topic_ko.strip())
        canonical_ko = hit if hit is not None else topic_ko
        canonical_en = reg_en.get(canonical_ko) or topic_en
        return canonical_ko, canonical_en, alias_miss

    return _fn


def _resolve_subtopic_via(topic_labels: set[str], modality: set[str]):
    """가짜 resolve_subtopic — 모달리티어/정본 topic 라벨이면 None(비움), 아니면 그대로."""

    def _fn(canonical_ko: str, subtopic_ko: str):
        if not subtopic_ko or not subtopic_ko.strip():
            return None
        if subtopic_ko.lower() in modality:
            return None
        if subtopic_ko in topic_labels:
            return None
        return subtopic_ko

    return _fn


class TestRewriteTopicRow(unittest.TestCase):
    def setUp(self):
        # 요리←음식 병합, 요리 en=cooking. 정본 topic 라벨 집합 {요리, 자전거}. 모달리티 {텍스트, 오디오}.
        self.rt = _resolve_topic_via({"음식": "요리", "요리": "요리", "자전거": "자전거"},
                                     {"요리": "cooking", "자전거": "bicycle"})
        self.rs = _resolve_subtopic_via({"요리", "자전거"}, {"텍스트", "오디오"})

    def test_alias_hit_rewrites_topic_to_canonical(self):
        old = {"topic_ko": "음식", "subtopic_ko": "김밥", "topic_en": "food", "subtopic_en": "gimbap"}
        new, changed, flags = rewrite_topic_row(old, self.rt, self.rs)
        self.assertEqual(new["topic_ko"], "요리")
        self.assertEqual(new["topic_en"], "cooking")
        self.assertEqual(new["subtopic_ko"], "김밥")  # 일반 subtopic 유지
        self.assertEqual(new["subtopic_en"], "gimbap")
        self.assertTrue(changed)
        self.assertTrue(flags["topic_changed"])
        self.assertFalse(flags["alias_miss"])

    def test_alias_miss_keeps_original_topic(self):
        old = {"topic_ko": "양자컴퓨팅", "subtopic_ko": "큐비트",
               "topic_en": "quantum", "subtopic_en": "qubit"}
        new, changed, flags = rewrite_topic_row(old, self.rt, self.rs)
        self.assertEqual(new["topic_ko"], "양자컴퓨팅")  # 미스 → 원본 유지(비파괴)
        self.assertEqual(new["topic_en"], "quantum")
        self.assertTrue(flags["alias_miss"])
        self.assertFalse(flags["topic_changed"])
        self.assertFalse(changed)  # subtopic 도 정상(비모달·비topic) → 변경 없음

    def test_topic_en_falls_back_when_registry_en_missing(self):
        # 정본 en 이 없으면(등록 en None) 기존 topic_en 보존(빈 라벨 방지).
        rt = _resolve_topic_via({"음식": "요리"}, {})  # reg_en 비어 → canonical_en=원본 en
        old = {"topic_ko": "음식", "subtopic_ko": "", "topic_en": "food", "subtopic_en": ""}
        new, _, _ = rewrite_topic_row(old, rt, self.rs)
        self.assertEqual(new["topic_ko"], "요리")
        self.assertEqual(new["topic_en"], "food")  # 정본 en 없음 → 원본 보존

    def test_subtopic_modality_cleared(self):
        old = {"topic_ko": "요리", "subtopic_ko": "텍스트", "topic_en": "cooking", "subtopic_en": "text"}
        new, changed, flags = rewrite_topic_row(old, self.rt, self.rs)
        self.assertEqual(new["subtopic_ko"], "")  # 모달리티어 → 비움
        self.assertEqual(new["subtopic_en"], "")  # en 도 함께 비움
        self.assertTrue(flags["subtopic_cleared"])
        self.assertTrue(changed)

    def test_subtopic_hierarchy_cleared(self):
        # subtopic 이 정본 topic 라벨(자전거)이면 계층 규칙으로 비움.
        old = {"topic_ko": "요리", "subtopic_ko": "자전거", "topic_en": "cooking", "subtopic_en": "bicycle"}
        new, changed, flags = rewrite_topic_row(old, self.rt, self.rs)
        self.assertEqual(new["subtopic_ko"], "")
        self.assertTrue(flags["subtopic_cleared"])

    def test_no_change_when_already_canonical(self):
        old = {"topic_ko": "요리", "subtopic_ko": "김밥", "topic_en": "cooking", "subtopic_en": "gimbap"}
        new, changed, flags = rewrite_topic_row(old, self.rt, self.rs)
        self.assertFalse(changed)
        self.assertEqual(new, {"topic_ko": "요리", "subtopic_ko": "김밥",
                               "topic_en": "cooking", "subtopic_en": "gimbap"})


class TestBuildPlanAndSC(unittest.TestCase):
    def setUp(self):
        self.rt = _resolve_topic_via({"음식": "요리", "요리": "요리", "등산": "등산", "여가": "여가"},
                                     {"요리": "cooking", "등산": "hiking", "여가": "leisure"})
        # 정본 topic 라벨 집합에 '등산' 포함 → subtopic '등산' 은 계층 규칙으로 비워짐(SC-03 후 0).
        self.rs = _resolve_subtopic_via({"요리", "등산", "여가"}, {"텍스트", "오디오", "영상", "이미지"})
        self.rows = [
            {"edge_id": "e1", "topic_ko": "음식", "subtopic_ko": "김밥",
             "topic_en": "food", "subtopic_en": "gimbap"},
            {"edge_id": "e2", "topic_ko": "여가", "subtopic_ko": "등산",  # subtopic=정본 topic → 비움
             "topic_en": "leisure", "subtopic_en": "hiking"},
            {"edge_id": "e3", "topic_ko": "요리", "subtopic_ko": "오디오",  # 모달리티 → 비움
             "topic_en": "cooking", "subtopic_en": "audio"},
        ]

    def test_build_plan_covers_all_rows(self):
        plan = build_plan(self.rows, self.rt, self.rs)
        self.assertEqual(len(plan), 3)
        self.assertEqual([p["edge_id"] for p in plan], ["e1", "e2", "e3"])

    def test_sc07_distinct_topics_shrinks(self):
        plan = build_plan(self.rows, self.rt, self.rs)
        news = [p["new"] for p in plan]
        # 음식→요리 병합: {요리, 여가} = 2 (원본은 음식/여가/요리 = 3)
        self.assertEqual(sc07_distinct_topics(news), 2)
        olds = [p["old"] for p in plan]
        self.assertEqual(sc07_distinct_topics(olds), 3)

    def test_sc03_overlap_zero_after(self):
        plan = build_plan(self.rows, self.rt, self.rs)
        news = [p["new"] for p in plan]
        # 재작성 후 topic∩subtopic 라벨 0 (등산 subtopic 비워짐)
        self.assertEqual(sc03_topic_subtopic_overlap(news), [])

    def test_sc04_modality_zero_after(self):
        plan = build_plan(self.rows, self.rt, self.rs)
        news = [p["new"] for p in plan]
        self.assertEqual(sc04_modality_subtopics(news), [])
        # 원본에는 오디오 모달리티 subtopic 존재
        olds = [p["old"] for p in plan]
        self.assertEqual(sc04_modality_subtopics(olds), ["오디오"])

    def test_summarize_plan_metrics(self):
        plan = build_plan(self.rows, self.rt, self.rs)
        rep = summarize_plan(plan)
        self.assertEqual(rep["n_edges"], 3)
        self.assertEqual(rep["sc07_before"], 3)
        self.assertEqual(rep["sc07_after"], 2)
        self.assertEqual(rep["sc03_after"], [])
        self.assertEqual(rep["sc04_after"], [])
        self.assertEqual(rep["alias_miss"], [])  # 전 topic 커버
        self.assertEqual(rep["n_subtopic_cleared"], 2)  # e2(등산)·e3(오디오)

    def test_format_report_lines_smoke(self):
        rep = summarize_plan(build_plan(self.rows, self.rt, self.rs))
        lines = format_report_lines(rep, mode="dry-run")
        self.assertTrue(any("SC-07" in ln for ln in lines))
        self.assertTrue(any("SC-03" in ln for ln in lines))
        self.assertTrue(any("SC-04" in ln for ln in lines))


class TestDbResolversLLMZero(unittest.TestCase):
    """DB 해소 결선(make_db_resolvers)이 alias 룩업·canonicalize_subtopic 만 쓰고 LLM/kNN/register 0 임을 단언."""

    def test_resolvers_use_only_lookup_seams_and_memoize(self):
        import src.relations.topic_canonicalize as tc

        with mock.patch.object(tc, "lookup_alias", return_value="요리") as m_alias, \
             mock.patch.object(tc, "_lookup_topic_en", return_value="cooking") as m_en, \
             mock.patch.object(tc, "canonicalize_subtopic", return_value="김밥") as m_sub, \
             mock.patch.object(tc, "knn_topic_candidates") as m_knn, \
             mock.patch.object(tc, "judge_topic") as m_judge, \
             mock.patch.object(tc, "register_topic") as m_reg, \
             mock.patch.object(tc, "canonicalize_topic") as m_cano:
            rt, rs = make_db_resolvers(conn=object())
            # 같은 키 2회 → 메모이즈로 seam 1회만
            self.assertEqual(rt("음식", "food"), ("요리", "cooking", False))
            self.assertEqual(rt("음식", "food"), ("요리", "cooking", False))
            self.assertEqual(rs("요리", "김밥"), "김밥")
            self.assertEqual(rs("요리", "김밥"), "김밥")

        m_alias.assert_called_once()  # 메모이즈
        m_en.assert_called_once()
        m_sub.assert_called_once()
        # LLM/kNN/register/canonicalize_topic 은 백필에서 절대 호출 안 함(결정성·LLM 0)
        m_knn.assert_not_called()
        m_judge.assert_not_called()
        m_reg.assert_not_called()
        m_cano.assert_not_called()

    def test_alias_miss_keeps_topic_via_db_resolver(self):
        import src.relations.topic_canonicalize as tc

        with mock.patch.object(tc, "lookup_alias", return_value=None), \
             mock.patch.object(tc, "_lookup_topic_en", return_value=None), \
             mock.patch.object(tc, "canonicalize_subtopic", return_value=None):
            rt, _ = make_db_resolvers(conn=object())
            canonical_ko, canonical_en, alias_miss = rt("양자컴퓨팅", "quantum")
        self.assertEqual(canonical_ko, "양자컴퓨팅")  # 미스 → 원본 유지
        self.assertEqual(canonical_en, "quantum")
        self.assertTrue(alias_miss)


if __name__ == "__main__":
    unittest.main()
