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

import unittest

from scripts.topic_grounding_report import (
    build_distribution_report,
    build_grounding_report,
    group_label_rows,
    pair_distribution,
    topic_distribution,
)


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


if __name__ == "__main__":
    unittest.main()
