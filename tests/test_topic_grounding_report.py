"""065 T402 — 자기주제 근거율(grounding) 리포트 순수 집계 단위 테스트(spec 065·FR-602·SC-02/03).

스펙 §개요의 근거 스크립트(자산 주제가 자기 summary/keywords 문자열에 등장하는지 = 보수적 하한)를
정식화한 리포트다. 실 DB 없이 순수 집계 규칙만 덮는다:
  - 분포(집계 A): topic_ko별·(topic_ko,subtopic_ko)별 카운트.
  - 부여율·미부여 사유(SC-03): 메타 보유(텍스트 있음) 대비 부여율 · no_text vs 분류실패 구분.
  - 근거율/오염율(집계 B·SC-02): 자산 주제가 자기 텍스트에 등장하면 grounded, 없으면 polluted.
  - (참고) 투영 방식과 동일 grounding 함수로 비교(다중 라벨 → 하나라도 근거 있으면 grounded).
DB 층(조회)은 사람 게이트(실 DB 실행) 몫 — 순수 함수만 여기서 단언한다.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.topic_grounding_report import (
    build_distribution_report,
    build_grounding_report,
    build_guard_report,
    compare_topic_smoke,
    group_label_rows,
    load_topic_smoke,
    pair_distribution,
    subtopic_concentration,
    topic_count_drift,
    topic_distribution,
    validate_topic_smoke,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _trows(*specs):
    """(asset_id, topic_ko, subtopic_ko) → asset_topic 행 dict 목록(테스트 헬퍼)."""
    return [{"asset_id": a, "topic_ko": t, "subtopic_ko": s} for a, t, s in specs]


# ────────────────────────────────────────────────────────────────────────────
# 분포(집계 A)
# ────────────────────────────────────────────────────────────────────────────
class TestDistribution(unittest.TestCase):
    def test_topic_counts(self):
        rows = _trows(
            ("a1", "스포츠·레저", "농구"),
            ("a2", "스포츠·레저", "축구"),
            ("a3", "과학", "천문"),
        )
        td = topic_distribution(rows)
        self.assertEqual(td["스포츠·레저"], 2)
        self.assertEqual(td["과학"], 1)

    def test_pair_counts(self):
        rows = _trows(
            ("a1", "스포츠·레저", "농구"),
            ("a2", "스포츠·레저", "농구"),
            ("a3", "스포츠·레저", "축구"),
        )
        pd = pair_distribution(rows)
        self.assertEqual(pd[("스포츠·레저", "농구")], 2)
        self.assertEqual(pd[("스포츠·레저", "축구")], 1)

    def test_empty(self):
        self.assertEqual(dict(topic_distribution([])), {})
        self.assertEqual(dict(pair_distribution([])), {})


# ────────────────────────────────────────────────────────────────────────────
# 부여율·미부여 사유(SC-03)
# ────────────────────────────────────────────────────────────────────────────
class TestDistributionReport(unittest.TestCase):
    def test_assignment_rate_and_missing_reasons(self):
        # a1: 텍스트 있고 주제 부여. a2: 텍스트 있으나 주제 없음(분류실패).
        # registered 5건 중 텍스트 보유 2건 → no_text = 3.
        topic_rows = _trows(("a1", "과학", "천문"))
        rep = build_distribution_report(
            topic_rows=topic_rows, text_asset_ids={"a1", "a2"}, n_registered=5
        )
        self.assertEqual(rep["n_with_topic"], 1)
        self.assertEqual(rep["n_no_text"], 3)  # 5 - 2
        self.assertEqual(rep["n_classify_failed"], 1)  # a2 텍스트 있는데 미부여
        self.assertEqual(rep["assignment_rate"], 0.5)  # 1/2 (텍스트 보유 대비)

    def test_zero_text_safe(self):
        rep = build_distribution_report(
            topic_rows=[], text_asset_ids=set(), n_registered=0
        )
        self.assertEqual(rep["assignment_rate"], 0.0)
        self.assertEqual(rep["n_classify_failed"], 0)


# ────────────────────────────────────────────────────────────────────────────
# 근거율/오염율(집계 B·SC-02)
# ────────────────────────────────────────────────────────────────────────────
class TestGrounding(unittest.TestCase):
    def test_subtopic_in_text_is_grounded(self):
        # 농구 골든: subtopic '농구'가 자기 텍스트에 등장 → grounded(오염 아님).
        rows = [
            {"asset_id": "a1", "labels": [("스포츠·레저", "농구")], "self_text": "1대1 농구대회 우승"}
        ]
        rep = build_grounding_report(rows)
        self.assertEqual(rep["n_grounded"], 1)
        self.assertEqual(rep["pollution_rate"], 0.0)

    def test_absent_label_is_polluted(self):
        # 자기 내용은 농구인데 주제가 배드민턴 → 근거 없음(오염).
        rows = [
            {"asset_id": "a1", "labels": [("스포츠·레저", "배드민턴")], "self_text": "1대1 농구대회"}
        ]
        rep = build_grounding_report(rows)
        self.assertEqual(rep["n_polluted"], 1)
        self.assertEqual(rep["pollution_rate"], 1.0)

    def test_topic_ko_match_also_grounds(self):
        rows = [
            {"asset_id": "a1", "labels": [("천문", None)], "self_text": "천문 관측 다큐"}
        ]
        rep = build_grounding_report(rows)
        self.assertEqual(rep["n_grounded"], 1)

    def test_multi_label_any_match_grounds(self):
        # 투영 스타일: 여러 라벨 중 하나라도 근거 있으면 grounded.
        rows = [
            {
                "asset_id": "a1",
                "labels": [("스포츠·레저", "축구"), ("스포츠·레저", "농구")],
                "self_text": "농구 경기 하이라이트",
            }
        ]
        rep = build_grounding_report(rows)
        self.assertEqual(rep["n_grounded"], 1)

    def test_empty_rows(self):
        rep = build_grounding_report([])
        self.assertEqual(rep["grounding_rate"], 0.0)
        self.assertEqual(rep["pollution_rate"], 0.0)
        self.assertEqual(rep["n_assets"], 0)


# ────────────────────────────────────────────────────────────────────────────
# 평탄 행 → 자산별 라벨 묶음(group_label_rows) — 정본/투영 공용
# ────────────────────────────────────────────────────────────────────────────
class TestGroupLabelRows(unittest.TestCase):
    def test_groups_multiple_labels_per_asset(self):
        flat = [
            {"asset_id": "a1", "topic_ko": "스포츠·레저", "subtopic_ko": "축구", "self_text": "농구"},
            {"asset_id": "a1", "topic_ko": "스포츠·레저", "subtopic_ko": "농구", "self_text": "농구"},
            {"asset_id": "a2", "topic_ko": "과학", "subtopic_ko": "천문", "self_text": "천문"},
        ]
        grouped = group_label_rows(flat)
        self.assertEqual(len(grouped), 2)
        by_id = {g["asset_id"]: g for g in grouped}
        self.assertEqual(
            set(by_id["a1"]["labels"]),
            {("스포츠·레저", "축구"), ("스포츠·레저", "농구")},
        )
        self.assertEqual(by_id["a1"]["self_text"], "농구")
        self.assertEqual(by_id["a2"]["labels"], [("과학", "천문")])


# ────────────────────────────────────────────────────────────────────────────
# 068 T401 — 분포 가드 지표(FR-301 · 집단 통계 불변식 · 순수 계산)
#   개별 자산 정답표가 아니라 집단 통계로 회귀를 감지한다(수동 학습化 방지).
#   가드는 계통 붕괴만 잡고 개별 애매는 수용한다.
# ────────────────────────────────────────────────────────────────────────────
def _many(topic, sub, n, prefix="a"):
    """같은 (topic, subtopic) 행 n개(테스트 헬퍼·고유 asset_id)."""
    return [{"asset_id": f"{prefix}{i}", "topic_ko": topic, "subtopic_ko": sub} for i in range(n)]


class TestSubtopicConcentration(unittest.TestCase):
    def test_dominant_real_subtopic_share(self):
        # 여행>관광지 과병합 재현: 8/10 이 관광지 → max_share 0.8, dominant '관광지'.
        rows = _many("여행·지역", "관광지", 8, "d")
        rows += _trows(("x1", "여행·지역", "도시여행"), ("x2", "여행·지역", "자연"))
        con = subtopic_concentration(rows)
        self.assertEqual(con["여행·지역"]["n_assets"], 10)
        self.assertEqual(con["여행·지역"]["dominant_subtopic"], "관광지")
        self.assertAlmostEqual(con["여행·지역"]["max_share"], 0.8)

    def test_none_and_misc_excluded_from_dominant(self):
        # None/기타 는 실 subtopic 아님 → dominant 후보에서 제외(분모는 topic 전체 자산).
        rows = _trows(
            ("a1", "과학", None), ("a2", "과학", "기타"),
            ("a3", "과학", "천문"), ("a4", "과학", "천문"),
        )
        con = subtopic_concentration(rows)
        self.assertEqual(con["과학"]["dominant_subtopic"], "천문")
        self.assertAlmostEqual(con["과학"]["max_share"], 0.5)  # 2/4(분모=전체)

    def test_no_real_subtopic_gives_zero_share(self):
        rows = _trows(("a1", "동물", None), ("a2", "동물", "기타"))
        con = subtopic_concentration(rows)
        self.assertIsNone(con["동물"]["dominant_subtopic"])
        self.assertEqual(con["동물"]["max_share"], 0.0)


class TestGuardReport(unittest.TestCase):
    def test_unassigned_rate_hard_violation(self):
        # registered 100 · 부여 80 → 미부여율 0.2 (>0.12 하드).
        rows = _many("과학", "천문", 80)
        rep = build_guard_report(topic_rows=rows, n_registered=100)
        self.assertAlmostEqual(rep["metrics"]["unassigned_rate"], 0.2)
        self.assertEqual(rep["metrics"]["n_unassigned"], 20)
        by_metric = {v["metric"]: v for v in rep["violations"]}
        self.assertIn("unassigned_rate", by_metric)
        self.assertEqual(by_metric["unassigned_rate"]["level"], "hard")
        self.assertEqual(rep["level"], "hard")

    def test_healthy_distribution_ok(self):
        # 균형 분산: 3개 실 subtopic(20/15/15) → 최대 점유율 0.4(<0.5 경보선). 미부여 0·기타 0.
        subs = ["천문"] * 20 + ["물리"] * 15 + ["화학"] * 15
        rows = [
            {"asset_id": f"a{i}", "topic_ko": "과학", "subtopic_ko": subs[i]}
            for i in range(50)
        ]
        rep = build_guard_report(topic_rows=rows, n_registered=50)
        self.assertAlmostEqual(rep["metrics"]["subtopic_concentration"]["과학"]["max_share"], 0.4)
        self.assertEqual(rep["level"], "ok")
        self.assertEqual(rep["violations"], [])

    def test_subtopic_concentration_hard_violation(self):
        # 관광지 45/50 = 0.9 → 하드(>0.7). n_assets>=표본하한.
        rows = _many("여행·지역", "관광지", 45, "d")
        rows += _many("여행·지역", "도시여행", 5, "c")
        rep = build_guard_report(topic_rows=rows, n_registered=50)
        conc = [v for v in rep["violations"] if v["metric"] == "subtopic_max_share"]
        self.assertEqual(len(conc), 1)
        self.assertEqual(conc[0]["scope"], "여행·지역")
        self.assertEqual(conc[0]["level"], "hard")
        self.assertEqual(conc[0]["dominant_subtopic"], "관광지")

    def test_small_topic_share_not_flagged(self):
        # 표본 과소 topic(3건 전부 같은 subtopic=1.0)은 점유율 노이즈라 미플래그.
        rows = _trows(("a1", "동물", "포유류"), ("a2", "동물", "포유류"), ("a3", "동물", "포유류"))
        rep = build_guard_report(topic_rows=rows, n_registered=3)
        conc = [v for v in rep["violations"] if v["metric"] == "subtopic_max_share"]
        self.assertEqual(conc, [])
        self.assertEqual(rep["level"], "ok")

    def test_misc_subtopic_rate_hard_violation(self):
        # 40/50 이 None/기타 → misc_rate 0.8 (>0.6 하드).
        rows = _many("생활·취미", None, 40, "n")
        rows += _many("생활·취미", "원예", 10, "g")
        rep = build_guard_report(topic_rows=rows, n_registered=50)
        m = [v for v in rep["violations"] if v["metric"] == "misc_subtopic_rate"]
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0]["level"], "hard")
        self.assertAlmostEqual(rep["metrics"]["misc_subtopic_rate"], 0.8)

    def test_singleton_pair_rate(self):
        # 실 pair 3종 중 2종이 싱글턴 → 0.6667.
        rows = _trows(
            ("a1", "과학", "천문"), ("a2", "과학", "천문"),
            ("a3", "과학", "물리"), ("a4", "과학", "화학"),
        )
        rep = build_guard_report(topic_rows=rows, n_registered=4)
        self.assertAlmostEqual(rep["metrics"]["singleton_pair_rate"], round(2 / 3, 4))

    def test_empty_safe(self):
        rep = build_guard_report(topic_rows=[], n_registered=0)
        self.assertEqual(rep["metrics"]["unassigned_rate"], 0.0)
        self.assertEqual(rep["metrics"]["singleton_pair_rate"], 0.0)
        self.assertEqual(rep["level"], "ok")

    def test_drift_included_when_before_given(self):
        rows = _many("과학", "천문", 8)
        rep = build_guard_report(
            topic_rows=rows, n_registered=8, before_topic_counts={"과학": 5, "동물": 3}
        )
        self.assertIn("drift", rep)
        self.assertEqual(rep["drift"]["per_topic"]["과학"]["delta"], 3)  # 8 - 5


class TestTopicCountDrift(unittest.TestCase):
    def test_deltas_and_churn(self):
        before = {"과학": 10, "여행·지역": 5}
        after = {"과학": 8, "여행·지역": 9, "동물": 2}
        d = topic_count_drift(before, after)
        self.assertEqual(d["per_topic"]["과학"]["delta"], -2)
        self.assertEqual(d["per_topic"]["여행·지역"]["delta"], 4)
        self.assertEqual(d["per_topic"]["동물"]["delta"], 2)  # 신규(before 0)
        self.assertEqual(d["total_churn"], 8)  # |−2|+|4|+|2|

    def test_empty(self):
        d = topic_count_drift({}, {})
        self.assertEqual(d["per_topic"], {})
        self.assertEqual(d["total_churn"], 0)


# ────────────────────────────────────────────────────────────────────────────
# 068 T402 — 고정 스모크셋(FR-302 · 6~8 앵커 · 늘리지 않음 · 재백필 후 대조)
# ────────────────────────────────────────────────────────────────────────────
class TestTopicSmokeGolden(unittest.TestCase):
    def test_load_shape_and_size(self):
        golden = load_topic_smoke()
        self.assertGreaterEqual(len(golden), 6)
        self.assertLessEqual(len(golden), 8)  # 늘리지 않음(수동 학습化 방지)
        for e in golden:
            self.assertIn("hint", e)
            self.assertIn("expected_topic", e)
            self.assertIn("expected_subtopic", e)
        # 무내용(검은배경) 앵커 = topic None(미부여) 존재.
        self.assertTrue(any(e["expected_topic"] is None for e in golden))

    def test_real_golden_is_valid(self):
        tax = json.loads((_REPO_ROOT / "src/relations/taxonomy_seed.json").read_text("utf-8"))
        valid = {t["topic_ko"] for t in tax["topics"] if t["topic_ko"] != "미분류"}
        problems = validate_topic_smoke(load_topic_smoke(), valid)
        self.assertEqual(problems, [], f"골든 스모크 무결성 위반: {problems}")


class TestCompareTopicSmoke(unittest.TestCase):
    def test_all_match(self):
        golden = [
            {"hint": "h1", "expected_topic": "스포츠·레저", "expected_subtopic": None},
            {"hint": "h2", "expected_topic": None, "expected_subtopic": None},
        ]
        actual = {
            "h1": {"topic_ko": "스포츠·레저", "subtopic_ko": "양궁"},  # subtopic 와일드카드
            "h2": {"topic_ko": None, "subtopic_ko": None},
        }
        rep = compare_topic_smoke(golden, actual)
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["topic_mismatches"], [])
        self.assertEqual(rep["n_topic_ok"], 2)

    def test_topic_mismatch(self):
        golden = [{"hint": "h1", "expected_topic": "IT·기술", "expected_subtopic": None}]
        actual = {"h1": {"topic_ko": "과학", "subtopic_ko": None}}
        rep = compare_topic_smoke(golden, actual)
        self.assertFalse(rep["passed"])
        self.assertEqual(len(rep["topic_mismatches"]), 1)
        self.assertEqual(rep["topic_mismatches"][0]["hint"], "h1")

    def test_none_topic_but_assigned_is_mismatch(self):
        # 무내용인데 topic 이 붙음 → 미부여 회귀(topic mismatch).
        golden = [{"hint": "h1", "expected_topic": None, "expected_subtopic": None}]
        actual = {"h1": {"topic_ko": "예술·공예", "subtopic_ko": None}}
        rep = compare_topic_smoke(golden, actual)
        self.assertEqual(len(rep["topic_mismatches"]), 1)

    def test_expected_subtopic_exact(self):
        golden = [{"hint": "h1", "expected_topic": "과학", "expected_subtopic": "천문"}]
        actual = {"h1": {"topic_ko": "과학", "subtopic_ko": "물리"}}
        rep = compare_topic_smoke(golden, actual)
        self.assertEqual(len(rep["subtopic_mismatches"]), 1)
        self.assertFalse(rep["passed"])

    def test_missing_actual_counts_as_mismatch(self):
        golden = [{"hint": "h1", "expected_topic": "과학", "expected_subtopic": None}]
        rep = compare_topic_smoke(golden, {})  # 실제 결과 없음
        self.assertEqual(len(rep["topic_mismatches"]), 1)

    def test_separation_violation(self):
        # 베네치아·풍차 가 같은 subtopic → 분리 위반(과병합).
        golden = [
            {"hint": "venice", "expected_topic": "여행·지역",
             "expected_subtopic": None, "distinct_from": ["windmill"]},
            {"hint": "windmill", "expected_topic": "여행·지역", "expected_subtopic": None},
        ]
        actual = {
            "venice": {"topic_ko": "여행·지역", "subtopic_ko": "관광지"},
            "windmill": {"topic_ko": "여행·지역", "subtopic_ko": "관광지"},
        }
        rep = compare_topic_smoke(golden, actual)
        self.assertEqual(len(rep["separation_violations"]), 1)
        self.assertFalse(rep["passed"])

    def test_separation_ok_when_distinct(self):
        golden = [
            {"hint": "venice", "expected_topic": "여행·지역",
             "expected_subtopic": None, "distinct_from": ["windmill"]},
            {"hint": "windmill", "expected_topic": "여행·지역", "expected_subtopic": None},
        ]
        actual = {
            "venice": {"topic_ko": "여행·지역", "subtopic_ko": "수변도시"},
            "windmill": {"topic_ko": "여행·지역", "subtopic_ko": "전원·시골"},
        }
        rep = compare_topic_smoke(golden, actual)
        self.assertEqual(rep["separation_violations"], [])
        self.assertTrue(rep["passed"])


class TestValidateTopicSmoke(unittest.TestCase):
    def test_unknown_topic_flagged(self):
        golden = [{"hint": "h1", "expected_topic": "없는토픽", "expected_subtopic": None}]
        problems = validate_topic_smoke(golden, {"과학", "여행·지역"})
        self.assertTrue(problems)

    def test_dangling_distinct_from_flagged(self):
        golden = [
            {"hint": "h1", "expected_topic": "과학",
             "expected_subtopic": None, "distinct_from": ["nope"]},
        ]
        problems = validate_topic_smoke(golden, {"과학"})
        self.assertTrue(problems)

    def test_valid_passes(self):
        golden = [
            {"hint": "h1", "expected_topic": "과학", "expected_subtopic": None},
            {"hint": "h2", "expected_topic": None, "expected_subtopic": None},
        ]
        self.assertEqual(validate_topic_smoke(golden, {"과학"}), [])


if __name__ == "__main__":
    unittest.main()
