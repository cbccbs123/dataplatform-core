"""미분류 감시 리포트 순수 집계 단위 테스트(spec 058 v2 · G13 · T1301 · FR-601v2).

거버넌스 §4(감시) — 닫힌 분류체계의 건강 지표를 주기 관측하는 **읽기전용** 리포트다.
실 DB/LLM 없이 순수 집계 규칙만 덮는다:
  - 미분류율(``unclassified_rate``): 미분류 엣지/전체 active 엣지 (거버넌스 §4 건강 지표).
  - 미분류 내 제안 라벨 누적(``unclassified_subtopics``): 미분류 엣지의 subtopic 빈도(범주 추가 근거).
  - 리포트 조립(``build_report``)·건강 임계 판정(healthy)·콘솔 포맷(``format_report_lines``).
DB 층(active 엣지 조회·미분류 자산 수·classify alias)은 e2e/수동 실행 몫(순수 함수만 여기서 단언).
"""
from __future__ import annotations

import unittest

from scripts.report_topic_unclassified import (
    build_report,
    format_report_lines,
    unclassified_rate,
    unclassified_subtopics,
)


def _rows(*specs):
    """(topic_ko, subtopic_ko) 튜플 목록 → 엣지 행 dict 목록(테스트 헬퍼)."""
    return [{"topic_ko": t, "subtopic_ko": s} for t, s in specs]


class TestUnclassifiedRate(unittest.TestCase):
    def test_counts_unclassified_over_total(self):
        rows = _rows(("음식·요리", "김밥"), ("미분류", "양자컴퓨팅"), ("스포츠·레저", "등산"))
        self.assertEqual(unclassified_rate(rows), (1, 3))

    def test_all_classified_zero(self):
        rows = _rows(("음식·요리", "김밥"), ("과학", "천문"))
        self.assertEqual(unclassified_rate(rows), (0, 2))

    def test_empty_rows(self):
        self.assertEqual(unclassified_rate([]), (0, 0))

    def test_does_not_count_etc_legacy_label(self):
        # catch-all 은 '미분류'(개명) — 옛 '기타'(guitar subtopic)는 미분류로 세지 않는다.
        rows = _rows(("음악", "기타"), ("미분류", "미지주제"))
        self.assertEqual(unclassified_rate(rows), (1, 2))


class TestUnclassifiedSubtopics(unittest.TestCase):
    def test_frequency_within_unclassified_only(self):
        rows = _rows(
            ("미분류", "양자컴퓨팅"),
            ("미분류", "양자컴퓨팅"),
            ("미분류", "블록체인"),
            ("과학", "천문"),  # 미분류 아님 → 제외
        )
        c = unclassified_subtopics(rows)
        self.assertEqual(dict(c), {"양자컴퓨팅": 2, "블록체인": 1})

    def test_blank_subtopic_ignored(self):
        rows = _rows(("미분류", ""), ("미분류", "   "), ("미분류", "블록체인"))
        self.assertEqual(dict(unclassified_subtopics(rows)), {"블록체인": 1})

    def test_no_unclassified_empty(self):
        rows = _rows(("음식·요리", "김밥"))
        self.assertEqual(dict(unclassified_subtopics(rows)), {})


class TestBuildReport(unittest.TestCase):
    def test_assembles_metrics_and_health(self):
        rows = _rows(
            ("음식·요리", "김밥"),
            ("미분류", "양자컴퓨팅"),
            ("미분류", "블록체인"),
            ("스포츠·레저", "등산"),
        )
        rep = build_report(
            rows, n_unclassified_assets=3, classify_raw_labels=["양자역학", "블록체인"]
        )
        self.assertEqual(rep["n_edges_total"], 4)
        self.assertEqual(rep["n_unclassified"], 2)
        self.assertEqual(rep["unclassified_rate"], 0.5)
        self.assertEqual(rep["n_unclassified_assets"], 3)
        self.assertEqual(
            rep["unclassified_subtopics"], {"양자컴퓨팅": 1, "블록체인": 1}
        )
        # classify alias 원본 라벨은 정렬(범주 추가 후보 목록).
        self.assertEqual(rep["classify_raw_labels"], ["블록체인", "양자역학"])
        # 미분류율 50% > 임계 → healthy False.
        self.assertFalse(rep["healthy"])

    def test_healthy_when_below_threshold(self):
        # 미분류 0/100 = 0% ≤ 5% → healthy True.
        rows = _rows(*([("음식·요리", "김밥")] * 100))
        rep = build_report(rows, n_unclassified_assets=0, classify_raw_labels=[])
        self.assertEqual(rep["unclassified_rate"], 0.0)
        self.assertTrue(rep["healthy"])

    def test_zero_edges_rate_zero(self):
        rep = build_report([], n_unclassified_assets=0, classify_raw_labels=[])
        self.assertEqual(rep["unclassified_rate"], 0.0)
        self.assertEqual(rep["n_edges_total"], 0)
        self.assertTrue(rep["healthy"])

    def test_custom_threshold(self):
        rows = _rows(("미분류", "x"), *([("음식·요리", "김밥")] * 9))  # 10% 미분류
        rep = build_report(
            rows, n_unclassified_assets=1, classify_raw_labels=[], threshold=0.2
        )
        self.assertTrue(rep["healthy"])  # 10% ≤ 20%
        rep2 = build_report(
            rows, n_unclassified_assets=1, classify_raw_labels=[], threshold=0.05
        )
        self.assertFalse(rep2["healthy"])  # 10% > 5%


class TestFormatReportLines(unittest.TestCase):
    def test_lines_contain_key_metrics(self):
        rep = build_report(
            _rows(("미분류", "양자컴퓨팅"), ("음식·요리", "김밥")),
            n_unclassified_assets=2,
            classify_raw_labels=["양자역학"],
        )
        text = "\n".join(format_report_lines(rep))
        self.assertIn("미분류율", text)
        self.assertIn("양자컴퓨팅", text)          # 제안 라벨 누적
        self.assertIn("양자역학", text)            # classify alias 후보
        self.assertIn("50.0%", text)               # 1/2

    def test_healthy_marker(self):
        rep = build_report(
            _rows(*([("음식·요리", "김밥")] * 20)),
            n_unclassified_assets=0,
            classify_raw_labels=[],
        )
        text = "\n".join(format_report_lines(rep))
        self.assertIn("건강", text)


if __name__ == "__main__":
    unittest.main()
