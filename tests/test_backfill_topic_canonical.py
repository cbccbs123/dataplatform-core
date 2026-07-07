"""백필 topic 정규화 스크립트 v2 단위 테스트(spec 058 v2 · G12 · T1201 · FR-501v2·C6·C7).

v2 백필은 **distinct (old_topic, old_subtopic) 쌍** 단위로 LLM 이 (범주 분류 + 주제어 subtopic 선정)을
결정해 캐시 동결하고 엣지를 재작성한다. 실 DB/LLM 없이 다음을 덮는다:
  - 후보 구성(``_candidate_sources``)·주제어 선정(``select_subtopic_term``·주입 select seam)의 순수 규칙
  - LLM 선정 seam(``_llm_select_subtopic``·complete_json mock)의 후보-강제·NONE 방어
  - 쌍→엣지 재작성 계획 조립(``build_plan``·주입 resolve_pair seam)·en 유도
  - SC 판정(topic∩subtopic·모달리티·distinct·기타율·분포)·리포트 포맷
  - DB 결선(``make_pair_resolver``)의 쌍 캐시(엣지수 아닌 쌍수만 LLM)
해소 seam(canonicalize_topic·select·canonicalize_subtopic)은 가짜 주입으로 조립·규칙만 검증한다.
"""
from __future__ import annotations

import unittest
from unittest import mock

from scripts.backfill_topic_canonical import (
    _candidate_sources,
    _llm_select_subtopic,
    build_plan,
    etc_rate,
    format_report_lines,
    make_pair_resolver,
    mapping_sample,
    sc03_topic_subtopic_overlap,
    sc04_modality_subtopics,
    sc07_distinct_topics,
    select_subtopic_term,
    summarize_plan,
    topic_distribution,
)


# ────────────────────────────────────────────────────────────────────────────
# 후보 구성(_candidate_sources) — 순수
# ────────────────────────────────────────────────────────────────────────────
class TestCandidateSources(unittest.TestCase):
    def test_both_present_distinct(self):
        # subtopic·topic 둘 다 후보(빈/모달 아님) — 각 출처 표식.
        srcs = _candidate_sources("요리", "김밥")
        self.assertEqual(srcs, {"김밥": "subtopic", "요리": "topic"})

    def test_modality_subtopic_excluded(self):
        srcs = _candidate_sources("요리", "텍스트")
        self.assertEqual(srcs, {"요리": "topic"})

    def test_modality_topic_excluded(self):
        srcs = _candidate_sources("영상", "폭포")
        self.assertEqual(srcs, {"폭포": "subtopic"})

    def test_empty_subtopic_only_topic(self):
        srcs = _candidate_sources("스포츠", "")
        self.assertEqual(srcs, {"스포츠": "topic"})

    def test_same_string_prefers_subtopic_source(self):
        srcs = _candidate_sources("등산", "등산")
        self.assertEqual(srcs, {"등산": "subtopic"})

    def test_all_empty(self):
        self.assertEqual(_candidate_sources("", ""), {})


# ────────────────────────────────────────────────────────────────────────────
# 주제어 선정(select_subtopic_term) — 순수(select seam 주입)
# ────────────────────────────────────────────────────────────────────────────
class TestSelectSubtopicTerm(unittest.TestCase):
    def test_no_candidates_returns_none_without_calling_llm(self):
        calls = []

        def _sel(nt, cands):
            calls.append((nt, cands))
            return "x"

        chosen, source = select_subtopic_term("음식·요리", "", "텍스트", select_fn=_sel)
        self.assertEqual((chosen, source), (None, None))
        self.assertEqual(calls, [])  # 후보 0 → LLM 미호출

    def test_picks_subtopic(self):
        # (요리, 김밥) → LLM 이 김밥 선택 → source=subtopic.
        chosen, source = select_subtopic_term(
            "음식·요리", "요리", "김밥", select_fn=lambda nt, cands: "김밥"
        )
        self.assertEqual((chosen, source), ("김밥", "subtopic"))

    def test_picks_topic(self):
        # (등산, 입문) → LLM 이 등산(옛 topic) 선택 → source=topic.
        chosen, source = select_subtopic_term(
            "스포츠·레저", "등산", "입문", select_fn=lambda nt, cands: "등산"
        )
        self.assertEqual((chosen, source), ("등산", "topic"))

    def test_single_candidate_still_offered_to_llm_can_reject(self):
        # (스포츠, "") 처럼 후보 1개(스포츠·범주급 광의어)여도 LLM 에 넘겨 거부(NONE) 가능해야 한다.
        seen = {}

        def _sel(nt, cands):
            seen["cands"] = cands
            return None  # LLM 이 광의어 거부

        chosen, source = select_subtopic_term("스포츠·레저", "스포츠", "", select_fn=_sel)
        self.assertEqual((chosen, source), (None, None))
        self.assertEqual(seen["cands"], ["스포츠"])  # 단일 후보도 LLM 에 제시

    def test_llm_returns_noncandidate_is_dropped(self):
        chosen, source = select_subtopic_term(
            "음식·요리", "요리", "김밥", select_fn=lambda nt, cands: "라면"
        )
        self.assertEqual((chosen, source), (None, None))

    def test_candidate_order_subtopic_first(self):
        # 후보 목록은 구체(subtopic) 우선으로 제시(광의 topic 뒤).
        seen = {}
        select_subtopic_term(
            "스포츠·레저", "등산", "보행법",
            select_fn=lambda nt, cands: seen.setdefault("c", cands) and None,
        )
        self.assertEqual(seen["c"], ["보행법", "등산"])


# ────────────────────────────────────────────────────────────────────────────
# LLM 선정 seam(_llm_select_subtopic) — complete_json mock
# ────────────────────────────────────────────────────────────────────────────
class TestLlmSelectSubtopic(unittest.TestCase):
    def test_valid_pick_in_candidates(self):
        with mock.patch("src.llm.client.complete_json", return_value={"subtopic": "김밥"}):
            self.assertEqual(_llm_select_subtopic("음식·요리", ["김밥", "요리"]), "김밥")

    def test_pick_not_in_candidates_returns_none(self):
        with mock.patch("src.llm.client.complete_json", return_value={"subtopic": "라면"}):
            self.assertIsNone(_llm_select_subtopic("음식·요리", ["김밥", "요리"]))

    def test_none_verdict_returns_none(self):
        with mock.patch("src.llm.client.complete_json", return_value={"subtopic": "NONE"}):
            self.assertIsNone(_llm_select_subtopic("스포츠·레저", ["스포츠"]))

    def test_empty_response_returns_none(self):
        with mock.patch("src.llm.client.complete_json", return_value={}):
            self.assertIsNone(_llm_select_subtopic("음식·요리", ["김밥"]))


# ────────────────────────────────────────────────────────────────────────────
# 쌍→엣지 재작성 계획(build_plan) — 순수(resolve_pair seam 주입)
# ────────────────────────────────────────────────────────────────────────────
def _fake_resolver(mapping: dict[tuple[str, str], dict]):
    """가짜 resolve_pair — (topic_ko, subtopic_ko) → 결정 dict(고정 매핑)."""

    def _fn(topic_ko: str, subtopic_ko: str) -> dict:
        return mapping[(topic_ko, subtopic_ko)]

    return _fn


class TestBuildPlanV2(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"edge_id": "e1", "topic_ko": "요리", "subtopic_ko": "김밥",
             "topic_en": "cooking", "subtopic_en": "gimbap"},
            {"edge_id": "e2", "topic_ko": "등산", "subtopic_ko": "입문",
             "topic_en": "hiking", "subtopic_en": "intro"},
            {"edge_id": "e3", "topic_ko": "스포츠", "subtopic_ko": "",
             "topic_en": "sports", "subtopic_en": ""},
        ]
        self.mapping = {
            # (요리,김밥): 옛 subtopic 김밥이 subtopic 유지 → source subtopic → en 은 subtopic_en.
            ("요리", "김밥"): {"new_topic_ko": "음식·요리", "new_topic_en": "food_cooking",
                              "new_subtopic_ko": "김밥", "selected_source": "subtopic"},
            # (등산,입문): 옛 topic 등산이 subtopic 로 → source topic → en 은 topic_en.
            ("등산", "입문"): {"new_topic_ko": "스포츠·레저", "new_topic_en": "sports_leisure",
                              "new_subtopic_ko": "등산", "selected_source": "topic"},
            # (스포츠,""): 광의어만 → subtopic None(비움).
            ("스포츠", ""): {"new_topic_ko": "스포츠·레저", "new_topic_en": "sports_leisure",
                            "new_subtopic_ko": None, "selected_source": None},
        }

    def test_assembles_new_topic_and_subtopic(self):
        plan = build_plan(self.rows, _fake_resolver(self.mapping))
        p1 = plan[0]
        self.assertEqual(p1["new"], {"topic_ko": "음식·요리", "subtopic_ko": "김밥",
                                     "topic_en": "food_cooking", "subtopic_en": "gimbap"})
        self.assertTrue(p1["changed"])
        self.assertTrue(p1["flags"]["topic_changed"])

    def test_old_topic_promoted_to_subtopic_uses_topic_en(self):
        plan = build_plan(self.rows, _fake_resolver(self.mapping))
        p2 = plan[1]
        # subtopic 이 옛 topic(등산)에서 왔으니 subtopic_en 은 옛 topic_en(hiking).
        self.assertEqual(p2["new"]["subtopic_ko"], "등산")
        self.assertEqual(p2["new"]["subtopic_en"], "hiking")
        self.assertEqual(p2["new"]["topic_ko"], "스포츠·레저")

    def test_subtopic_stays_empty_when_decision_none(self):
        # e3 (스포츠, "") → 광의어라 subtopic None. 원래 빈값이라 비움은 아니고 topic 만 변경.
        plan = build_plan(self.rows, _fake_resolver(self.mapping))
        p3 = plan[2]
        self.assertEqual(p3["new"]["subtopic_ko"], "")
        self.assertEqual(p3["new"]["subtopic_en"], "")
        self.assertFalse(p3["flags"]["subtopic_changed"])  # "" → "" 변경 없음
        self.assertTrue(p3["flags"]["topic_changed"])

    def test_nonempty_subtopic_cleared_when_decision_none(self):
        # 비지 않은 subtopic 이 결정 None(모달/범용)이면 비움 플래그.
        rows = [{"edge_id": "z", "topic_ko": "배경", "subtopic_ko": "텍스트",
                 "topic_en": "bg", "subtopic_en": "text"}]
        mapping = {("배경", "텍스트"): {"new_topic_ko": "미분류", "new_topic_en": "unclassified",
                                      "new_subtopic_ko": None, "selected_source": None}}
        plan = build_plan(rows, _fake_resolver(mapping))
        self.assertEqual(plan[0]["new"]["subtopic_ko"], "")
        self.assertTrue(plan[0]["flags"]["subtopic_cleared"])

    def test_etc_flag_when_topic_is_etc(self):
        rows = [{"edge_id": "x", "topic_ko": "배경", "subtopic_ko": "",
                 "topic_en": "bg", "subtopic_en": ""}]
        mapping = {("배경", ""): {"new_topic_ko": "미분류", "new_topic_en": "unclassified",
                                 "new_subtopic_ko": None, "selected_source": None}}
        plan = build_plan(rows, _fake_resolver(mapping))
        self.assertTrue(plan[0]["flags"]["etc_topic"])


# ────────────────────────────────────────────────────────────────────────────
# SC 판정·분포·리포트 — 순수
# ────────────────────────────────────────────────────────────────────────────
class TestScAndReport(unittest.TestCase):
    def _plan(self):
        rows = [
            {"edge_id": "e1", "topic_ko": "요리", "subtopic_ko": "김밥",
             "topic_en": "cooking", "subtopic_en": "gimbap"},
            {"edge_id": "e2", "topic_ko": "등산", "subtopic_ko": "입문",
             "topic_en": "hiking", "subtopic_en": "intro"},
            {"edge_id": "e3", "topic_ko": "스포츠", "subtopic_ko": "",
             "topic_en": "sports", "subtopic_en": ""},
            {"edge_id": "e4", "topic_ko": "배경", "subtopic_ko": "텍스트",
             "topic_en": "bg", "subtopic_en": "text"},
        ]
        mapping = {
            ("요리", "김밥"): {"new_topic_ko": "음식·요리", "new_topic_en": "food_cooking",
                              "new_subtopic_ko": "김밥", "selected_source": "subtopic"},
            ("등산", "입문"): {"new_topic_ko": "스포츠·레저", "new_topic_en": "sports_leisure",
                              "new_subtopic_ko": "등산", "selected_source": "topic"},
            ("스포츠", ""): {"new_topic_ko": "스포츠·레저", "new_topic_en": "sports_leisure",
                            "new_subtopic_ko": None, "selected_source": None},
            ("배경", "텍스트"): {"new_topic_ko": "미분류", "new_topic_en": "unclassified",
                               "new_subtopic_ko": None, "selected_source": None},
        }
        return build_plan(rows, _fake_resolver(mapping))

    def test_sc07_distinct_new_topics(self):
        news = [p["new"] for p in self._plan()]
        # 음식·요리, 스포츠·레저(x2), 기타 = 3 distinct.
        self.assertEqual(sc07_distinct_topics(news), 3)

    def test_sc03_no_overlap_after(self):
        news = [p["new"] for p in self._plan()]
        self.assertEqual(sc03_topic_subtopic_overlap(news), [])

    def test_sc04_no_modality_after(self):
        news = [p["new"] for p in self._plan()]
        self.assertEqual(sc04_modality_subtopics(news), [])
        # 원본엔 텍스트 모달리티 subtopic 존재.
        olds = [p["old"] for p in self._plan()]
        self.assertEqual(sc04_modality_subtopics(olds), ["텍스트"])

    def test_topic_distribution(self):
        news = [p["new"] for p in self._plan()]
        dist = topic_distribution(news)
        self.assertEqual(dist["스포츠·레저"], 2)
        self.assertEqual(dist["음식·요리"], 1)
        self.assertEqual(dist["미분류"], 1)

    def test_etc_rate(self):
        news = [p["new"] for p in self._plan()]
        n_etc, n_total = etc_rate(news)
        self.assertEqual((n_etc, n_total), (1, 4))

    def test_summarize_and_format(self):
        rep = summarize_plan(self._plan())
        self.assertEqual(rep["n_edges"], 4)
        self.assertEqual(rep["sc07_after"], 3)
        self.assertEqual(rep["sc03_after"], [])
        self.assertEqual(rep["sc04_after"], [])
        self.assertEqual(rep["n_etc"], 1)
        self.assertEqual(rep["n_pairs"], 4)
        lines = format_report_lines(rep, mode="dry-run")
        self.assertTrue(any("topic∩subtopic" in ln for ln in lines))
        self.assertTrue(any("미분류율" in ln for ln in lines))
        self.assertTrue(any("분포" in ln for ln in lines))

    def test_mapping_sample_distinct_pairs(self):
        sample = mapping_sample(self._plan(), n=20)
        # distinct 쌍 4개 · old→new 표기 포함.
        self.assertEqual(len(sample), 4)
        keys = {(s["old_topic"], s["old_subtopic"]) for s in sample}
        self.assertIn(("요리", "김밥"), keys)
        one = next(s for s in sample if s["old_topic"] == "요리")
        self.assertEqual(one["new_topic"], "음식·요리")
        self.assertEqual(one["new_subtopic"], "김밥")
        self.assertEqual(one["count"], 1)


# ────────────────────────────────────────────────────────────────────────────
# DB 결선 쌍 캐시(make_pair_resolver) — 엣지수 아닌 쌍수만 LLM
# ────────────────────────────────────────────────────────────────────────────
class TestMakePairResolver(unittest.TestCase):
    def test_pair_cache_computes_once(self):
        import scripts.backfill_topic_canonical as bf

        with mock.patch.object(
            bf, "canonicalize_topic",
            return_value={"canonical_ko": "음식·요리", "canonical_en": "food_cooking",
                          "decided_by": "classify"},
        ) as m_topic, \
            mock.patch.object(bf, "_llm_select_subtopic", return_value="김밥") as m_sel, \
            mock.patch.object(bf, "canonicalize_subtopic", return_value="김밥") as m_sub:
            resolve, stats = bf.make_pair_resolver(conn=object())
            d1 = resolve("요리", "김밥")
            d2 = resolve("요리", "김밥")  # 같은 쌍 → 캐시(재계산 0)

        self.assertEqual(d1, d2)
        self.assertEqual(d1["new_topic_ko"], "음식·요리")
        self.assertEqual(d1["new_subtopic_ko"], "김밥")
        self.assertEqual(d1["selected_source"], "subtopic")
        # 같은 쌍 2회 → 각 seam 1회만(쌍 캐시).
        m_topic.assert_called_once()
        m_sel.assert_called_once()
        m_sub.assert_called_once()
        self.assertEqual(stats["n_pairs"], 1)

    def test_subtopic_none_skips_canonicalize_subtopic(self):
        import scripts.backfill_topic_canonical as bf

        with mock.patch.object(
            bf, "canonicalize_topic",
            return_value={"canonical_ko": "스포츠·레저", "canonical_en": "sports_leisure",
                          "decided_by": "classify"},
        ), \
            mock.patch.object(bf, "_llm_select_subtopic", return_value=None) as m_sel, \
            mock.patch.object(bf, "canonicalize_subtopic") as m_sub:
            resolve, _ = bf.make_pair_resolver(conn=object())
            d = resolve("스포츠", "")

        self.assertIsNone(d["new_subtopic_ko"])
        # (스포츠,"") → 후보 1개(스포츠)라 LLM 선정은 호출되나 광의어 거부(None) →
        # canonicalize_subtopic 은 건너뛴다(주제어 없음·불필요한 등록/kNN 방지).
        m_sel.assert_called_once()
        m_sub.assert_not_called()


if __name__ == "__main__":
    unittest.main()
