"""주제 투영 함수 ``project_asset_topics`` 단위 테스트 (mock, DB·LLM 불필요).

검증 의도 (056 G1 · FR-101~105·FR-601·SC-01)
    - ``project_asset_topics`` 는 ``graph_query.fetch_active_relations_for_asset(status='active')``
      의 이웃 ``topic`` 을 ``(topic_ko, subtopic_ko)`` 로 그룹·집계(weight = 이웃 수)한다.
    - seam 재사용: ``fetch_active_relations_for_asset`` 를 **topic_query 모듈 위치에서** patch 해
      실 DB 없이 이웃 목록을 통제한다(``src.relations.topic_query.fetch_active_relations_for_asset``).
    - 빈/None ``topic_ko`` 또는 dict 아닌 ``topic`` 이웃은 스킵(주제 미부여 엣지).
    - 결정성(헌법 3조): 정렬 타이브레이커 ``weight desc → topic_ko asc → subtopic_ko asc`` 후 top_n 절단.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

_PATCH_TARGET = "src.relations.topic_query.fetch_active_relations_for_asset"


def _conn():
    """fetch_active_relations_for_asset 를 patch 하므로 conn 은 통과용 센티널이면 충분."""
    return object()


def _nb(topic, **over):
    """이웃 dict 한 개(graph_query 반환 형상의 부분집합). 투영은 ``topic`` 만 사용."""
    base = {"asset_id": "X", "kind_code": "duplicate_near", "topic": topic}
    base.update(over)
    return base


def _topic(topic_ko, subtopic_ko, topic_en, subtopic_en):
    return {
        "topic_ko": topic_ko,
        "subtopic_ko": subtopic_ko,
        "topic_en": topic_en,
        "subtopic_en": subtopic_en,
    }


class TestProjectAssetTopics(unittest.TestCase):
    """T101 — 투영·집계·스킵·결정적 정렬·top_n·반환 형상."""

    @patch(_PATCH_TARGET)
    def test_three_neighbors_same_topic_weight_three(self, m_fetch) -> None:
        # 같은 (topic_ko, subtopic_ko) 이웃 3개 → 결과 1건, weight 3
        from src.relations.topic_query import project_asset_topics

        t = _topic("요리", "제빵", "cooking", "baking")
        conn = _conn()
        m_fetch.return_value = [_nb(dict(t)), _nb(dict(t)), _nb(dict(t))]

        out = project_asset_topics(conn, asset_id="A")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["weight"], 3)
        self.assertEqual(out[0]["topic_ko"], "요리")
        self.assertEqual(out[0]["subtopic_ko"], "제빵")
        # seam 재사용: conn 통과 + asset_id + active-only 로 호출
        m_fetch.assert_called_once()
        args, kwargs = m_fetch.call_args
        self.assertIs(args[0], conn)
        self.assertEqual(kwargs.get("asset_id"), "A")
        self.assertEqual(kwargs.get("status"), "active")

    @patch(_PATCH_TARGET)
    def test_empty_or_none_topic_ko_and_nondict_skipped(self, m_fetch) -> None:
        # 빈 topic_ko·None topic_ko·dict 아닌 topic(None) 은 스킵(집계 제외)
        from src.relations.topic_query import project_asset_topics

        good = _topic("요리", "제빵", "cooking", "baking")
        m_fetch.return_value = [
            _nb(dict(good)),
            _nb({"topic_ko": "", "subtopic_ko": "x"}),      # 빈 문자열 → 스킵
            _nb({"topic_ko": None, "subtopic_ko": "y"}),    # None → 스킵
            _nb(None),                                       # topic 자체 None(비-dict) → 스킵
        ]

        out = project_asset_topics(_conn(), asset_id="A")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["topic_ko"], "요리")
        self.assertEqual(out[0]["weight"], 1)

    @patch(_PATCH_TARGET)
    def test_two_distinct_topics_aggregate_separately(self, m_fetch) -> None:
        # 서로 다른 두 주제는 별도 엔트리로 집계
        from src.relations.topic_query import project_asset_topics

        t1 = _topic("요리", "제빵", "cooking", "baking")
        t2 = _topic("음악", "재즈", "music", "jazz")
        m_fetch.return_value = [_nb(dict(t1)), _nb(dict(t2)), _nb(dict(t1))]

        out = project_asset_topics(_conn(), asset_id="A")

        self.assertEqual(len(out), 2)
        weights = {(e["topic_ko"], e["subtopic_ko"]): e["weight"] for e in out}
        self.assertEqual(weights[("요리", "제빵")], 2)
        self.assertEqual(weights[("음악", "재즈")], 1)

    @patch(_PATCH_TARGET)
    def test_top_n_keeps_highest_weight_desc(self, m_fetch) -> None:
        # 서로 다른 3주제(weight 3/2/1) + top_n=2 → weight desc 상위 2건
        from src.relations.topic_query import project_asset_topics

        hi = _topic("요리", "제빵", "cooking", "baking")     # weight 3
        mid = _topic("음악", "재즈", "music", "jazz")         # weight 2
        lo = _topic("여행", "국내", "travel", "domestic")     # weight 1
        m_fetch.return_value = (
            [_nb(dict(hi))] * 3 + [_nb(dict(mid))] * 2 + [_nb(dict(lo))]
        )

        out = project_asset_topics(_conn(), asset_id="A", top_n=2)

        self.assertEqual(len(out), 2)
        self.assertEqual([e["weight"] for e in out], [3, 2])
        self.assertEqual([e["topic_ko"] for e in out], ["요리", "음악"])

    @patch(_PATCH_TARGET)
    def test_tiebreak_topic_ko_then_subtopic_ko_asc(self, m_fetch) -> None:
        # weight 동점(모두 1) → topic_ko asc, 같은 topic_ko 내 subtopic_ko asc (결정성)
        from src.relations.topic_query import project_asset_topics

        m_fetch.return_value = [
            _nb(_topic("요리", "제빵", "cooking", "baking")),
            _nb(_topic("요리", "제과", "cooking", "confectionery")),
            _nb(_topic("음악", "재즈", "music", "jazz")),
        ]

        out = project_asset_topics(_conn(), asset_id="A")

        keys = [(e["topic_ko"], e["subtopic_ko"]) for e in out]
        # 요리 < 음악 (topic_ko asc); 요리 내부 제과 < 제빵 (subtopic_ko asc)
        self.assertEqual(keys, [("요리", "제과"), ("요리", "제빵"), ("음악", "재즈")])

    @patch(_PATCH_TARGET)
    def test_entry_shape_exact_keys(self, m_fetch) -> None:
        # 반환 각 엔트리 형상 == {topic_ko, subtopic_ko, topic_en, subtopic_en, weight}
        from src.relations.topic_query import project_asset_topics

        m_fetch.return_value = [_nb(_topic("요리", "제빵", "cooking", "baking"))]

        out = project_asset_topics(_conn(), asset_id="A")

        self.assertEqual(
            set(out[0].keys()),
            {"topic_ko", "subtopic_ko", "topic_en", "subtopic_en", "weight"},
        )
        self.assertEqual(out[0]["topic_en"], "cooking")
        self.assertEqual(out[0]["subtopic_en"], "baking")

    @patch(_PATCH_TARGET)
    def test_deterministic_same_input_same_output(self, m_fetch) -> None:
        # 같은 입력 2회 → 같은 출력(헌법 3조). 투영은 순수.
        from src.relations.topic_query import project_asset_topics

        rows = [
            _nb(_topic("요리", "제빵", "cooking", "baking")),
            _nb(_topic("음악", "재즈", "music", "jazz")),
            _nb(_topic("요리", "제빵", "cooking", "baking")),
        ]
        m_fetch.return_value = [dict(r) for r in rows]
        first = project_asset_topics(_conn(), asset_id="A")
        m_fetch.return_value = [dict(r) for r in rows]
        second = project_asset_topics(_conn(), asset_id="A")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
