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
from unittest.mock import MagicMock, patch

_PATCH_TARGET = "src.relations.topic_query.fetch_active_relations_for_asset"


def _conn():
    """fetch_active_relations_for_asset 를 patch 하므로 conn 은 통과용 센티널이면 충분."""
    return object()


def _mock_conn(rows):
    """``conn.cursor(row_factory=dict_row)`` 컨텍스트매니저를 흉내내는 mock conn.

    graph_query/review 단위 테스트와 동형(``tests/test_graph_query.py`` 의 ``_conn_returning``):
    ``__enter__`` 가 cur 를 돌려주고 ``fetchall`` 이 주입 행을 반환한다. ``execute`` 인자는
    ``cur.execute.call_args`` 로 캡처해 SQL 부분문자열(의료 제외·topic 표현식)·바인딩을 검증한다.
    """
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchall.return_value = rows
    conn.cursor.return_value = cur
    return conn, cur


def _compact_sql(cur):
    """마지막 execute 의 SQL 을 공백 정규화해 견고한 부분문자열 검사용으로 반환."""
    return " ".join(cur.execute.call_args[0][0].split())


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


class TestFindTopicNeighbors(unittest.TestCase):
    """T201 — 같은-주제 이웃: overlap_weight 정렬·already_linked·str 강제·top_k·의료 제외 조건."""

    @patch(_PATCH_TARGET)
    def test_overlap_sort_shared_topics_and_already_linked(self, m_fetch) -> None:
        from src.relations.topic_query import find_topic_neighbors

        # m_fetch 는 project_asset_topics(대상 주제) + already_linked(직접 이웃) 두 용도로 공유된다.
        # 대상 A 의 직접 이웃 = B(주제 '요리') → 대상 주제 {'요리'}, linked 집합 {'B'}.
        m_fetch.return_value = [_nb(_topic("요리", "제빵", "cooking", "baking"), asset_id="B")]

        # '요리' 주제를 실은 active 엣지들(cursor 반환). 양끝 자산을 후보로 수집한다.
        rows = [
            {"src_asset": "A", "dst_asset": "B", "topic_ko": "요리"},   # B +1 (A 는 대상→스킵)
            {"src_asset": "C", "dst_asset": "D", "topic_ko": "요리"},   # C +1, D +1
            {"src_asset": "C", "dst_asset": "E", "topic_ko": "요리"},   # C +1(=2), E +1
        ]
        conn, cur = _mock_conn(rows)

        out = find_topic_neighbors(conn, asset_id="A", top_k=10)

        # 대상 A 자신은 결과에서 제외
        self.assertNotIn("A", [o["asset_id"] for o in out])
        # overlap_weight desc → asset_id asc: C(2) > B(1),D(1),E(1) → C,B,D,E
        self.assertEqual([o["asset_id"] for o in out], ["C", "B", "D", "E"])
        self.assertEqual(out[0]["overlap_weight"], 2)
        # already_linked: B 는 대상 직접 이웃(True), 나머지 False
        linked = {o["asset_id"]: o["already_linked"] for o in out}
        self.assertTrue(linked["B"])
        self.assertFalse(linked["C"])
        self.assertFalse(linked["D"])
        # shared_topics 는 공유 topic_ko 집합(정렬)
        self.assertEqual(out[0]["shared_topics"], ["요리"])
        # 엔트리 형상
        self.assertEqual(
            set(out[0].keys()),
            {"asset_id", "shared_topics", "overlap_weight", "already_linked"},
        )

    @patch(_PATCH_TARGET)
    def test_asset_id_coerced_to_str(self, m_fetch) -> None:
        # 조회행 계약(graph_query 관례): asset_id 반환은 항상 str. 비-str 입력(UUID/int)도 강제.
        import uuid

        from src.relations.topic_query import find_topic_neighbors

        m_fetch.return_value = [_nb(_topic("요리", "제빵", "cooking", "baking"), asset_id="A")]
        u = uuid.uuid4()
        rows = [{"src_asset": "A", "dst_asset": u, "topic_ko": "요리"}]
        conn, _ = _mock_conn(rows)

        out = find_topic_neighbors(conn, asset_id="A")

        self.assertEqual(len(out), 1)
        self.assertIsInstance(out[0]["asset_id"], str)
        self.assertEqual(out[0]["asset_id"], str(u))

    @patch(_PATCH_TARGET)
    def test_top_k_truncates_after_sort(self, m_fetch) -> None:
        from src.relations.topic_query import find_topic_neighbors

        m_fetch.return_value = [_nb(_topic("요리", "제빵", "cooking", "baking"), asset_id="Z")]
        # C overlap 2, 나머지 1 → top_k=2 면 C + (asset_id asc 최솟값) 만.
        rows = [
            {"src_asset": "C", "dst_asset": "D", "topic_ko": "요리"},
            {"src_asset": "C", "dst_asset": "E", "topic_ko": "요리"},
            {"src_asset": "F", "dst_asset": "G", "topic_ko": "요리"},
        ]
        conn, _ = _mock_conn(rows)

        out = find_topic_neighbors(conn, asset_id="A", top_k=2)

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["asset_id"], "C")   # overlap 2
        self.assertEqual(out[1]["asset_id"], "D")   # overlap 1, asset_id asc 최솟값

    @patch(_PATCH_TARGET)
    def test_no_target_topics_returns_empty_without_query(self, m_fetch) -> None:
        # 대상에 주제가 없으면(직접 이웃의 topic 미부여) 빈 리스트 — cursor 조회조차 안 한다.
        from src.relations.topic_query import find_topic_neighbors

        m_fetch.return_value = [_nb(None, asset_id="B")]  # topic 미부여 → 대상 주제 {}
        conn, cur = _mock_conn([])

        out = find_topic_neighbors(conn, asset_id="A")

        self.assertEqual(out, [])
        cur.execute.assert_not_called()

    @patch(_PATCH_TARGET)
    def test_sql_has_medical_exclusion_and_topic_index_predicate(self, m_fetch) -> None:
        # 의료 제외 조건(review._build_review_where 재사용) + 표현식 인덱스 친화 술어 + 바인딩.
        from src.relations.topic_query import find_topic_neighbors

        m_fetch.return_value = [_nb(_topic("요리", "제빵", "cooking", "baking"), asset_id="B")]
        conn, cur = _mock_conn([])

        find_topic_neighbors(conn, asset_id="A")

        sql = _compact_sql(cur)
        self.assertIn("sa.domain_label IS DISTINCT FROM 'medical'", sql)
        self.assertIn("da.domain_label IS DISTINCT FROM 'medical'", sql)
        self.assertIn("ge.status = 'active'", sql)
        # 표현식 인덱스(ix_graph_edge_topic_ko) 친화 술어 + ANY 바인딩(f-string 값 주입 금지)
        self.assertIn("ge.topic->>'topic_ko' = ANY(%s)", sql)
        params = cur.execute.call_args[0][1]
        self.assertEqual(params[0], ["요리"])   # 대상 주제 topic_ko 리스트(결정적 정렬)


class TestListTopics(unittest.TestCase):
    """T201 — 주제 목록: (topic_ko, subtopic_ko) 별 distinct 양끝 자산 수·정렬·의료 제외."""

    def test_distinct_endpoint_assets_per_topic_and_sort(self) -> None:
        from src.relations.topic_query import list_topics

        rows = [
            {"src_asset": "A", "dst_asset": "B", "topic_ko": "요리", "subtopic_ko": "제빵"},
            {"src_asset": "B", "dst_asset": "C", "topic_ko": "요리", "subtopic_ko": "제빵"},
            {"src_asset": "D", "dst_asset": "E", "topic_ko": "음악", "subtopic_ko": "재즈"},
            {"src_asset": "F", "dst_asset": "G", "topic_ko": "요리", "subtopic_ko": ""},
        ]
        conn, _ = _mock_conn(rows)

        out = list_topics(conn)

        # 정렬: topic_ko asc → subtopic_ko asc(None/"" 은 "" 로 최상단). 요리<음악.
        keys = [(o["topic_ko"], o["subtopic_ko"]) for o in out]
        self.assertEqual(keys, [("요리", None), ("요리", "제빵"), ("음악", "재즈")])
        counts = {(o["topic_ko"], o["subtopic_ko"]): o["asset_count"] for o in out}
        self.assertEqual(counts[("요리", "제빵")], 3)   # {A,B,C} distinct
        self.assertEqual(counts[("음악", "재즈")], 2)   # {D,E}
        self.assertEqual(counts[("요리", None)], 2)     # {F,G}, 빈 subtopic → None 정규화

    def test_empty_topic_ko_rows_skipped(self) -> None:
        from src.relations.topic_query import list_topics

        rows = [
            {"src_asset": "A", "dst_asset": "B", "topic_ko": "요리", "subtopic_ko": "제빵"},
            {"src_asset": "C", "dst_asset": "D", "topic_ko": "", "subtopic_ko": "x"},   # 스킵
            {"src_asset": "E", "dst_asset": "F", "topic_ko": None, "subtopic_ko": "y"}, # 스킵
        ]
        conn, _ = _mock_conn(rows)

        out = list_topics(conn)

        self.assertEqual([(o["topic_ko"], o["subtopic_ko"]) for o in out], [("요리", "제빵")])

    def test_sql_has_medical_exclusion(self) -> None:
        from src.relations.topic_query import list_topics

        conn, cur = _mock_conn([])
        list_topics(conn)

        sql = _compact_sql(cur)
        self.assertIn("sa.domain_label IS DISTINCT FROM 'medical'", sql)
        self.assertIn("da.domain_label IS DISTINCT FROM 'medical'", sql)
        self.assertIn("ge.status = 'active'", sql)


class TestAssetsInTopic(unittest.TestCase):
    """T201 — 주제별 자산 페이징: distinct 양끝·asset_id asc 결정적·total·의료 제외·subtopic 필터."""

    @staticmethod
    def _rows():
        return [
            {"src_asset": "a3", "src_fs_uri": None, "src_fs_path": "/x/a3.txt",
             "dst_asset": "a1", "dst_fs_uri": "uri1", "dst_fs_path": "/x/a1.txt"},
            {"src_asset": "a2", "src_fs_uri": None, "src_fs_path": "/x/a2.mp4",
             "dst_asset": "a1", "dst_fs_uri": "uri1", "dst_fs_path": "/x/a1.txt"},
        ]

    def test_distinct_assets_sorted_and_total(self) -> None:
        from src.relations.topic_query import assets_in_topic

        conn, _ = _mock_conn(self._rows())

        out = assets_in_topic(conn, topic_ko="요리")

        # distinct 자산 {a1,a2,a3} → asset_id asc, total=3
        self.assertEqual(out["total"], 3)
        self.assertEqual([r["asset_id"] for r in out["rows"]], ["a1", "a2", "a3"])
        # 식별 필드: fs_uri + file_name(basename)
        first = out["rows"][0]
        self.assertEqual(first["file_name"], "a1.txt")
        self.assertEqual(first["fs_uri"], "uri1")

    def test_pagination_limit_offset(self) -> None:
        from src.relations.topic_query import assets_in_topic

        conn, _ = _mock_conn(self._rows())

        page1 = assets_in_topic(conn, topic_ko="요리", limit=2, offset=0)
        self.assertEqual([r["asset_id"] for r in page1["rows"]], ["a1", "a2"])
        self.assertEqual(page1["total"], 3)

        page2 = assets_in_topic(conn, topic_ko="요리", limit=2, offset=2)
        self.assertEqual([r["asset_id"] for r in page2["rows"]], ["a3"])
        self.assertEqual(page2["total"], 3)

    def test_asset_id_coerced_to_str(self) -> None:
        import uuid

        from src.relations.topic_query import assets_in_topic

        u = uuid.uuid4()
        rows = [{"src_asset": u, "src_fs_uri": None, "src_fs_path": "/x/u.txt",
                 "dst_asset": "z", "dst_fs_uri": None, "dst_fs_path": "/x/z.txt"}]
        conn, _ = _mock_conn(rows)

        out = assets_in_topic(conn, topic_ko="요리")

        ids = [r["asset_id"] for r in out["rows"]]
        self.assertTrue(all(isinstance(i, str) for i in ids))
        self.assertIn(str(u), ids)

    def test_subtopic_filter_adds_predicate_and_binding(self) -> None:
        from src.relations.topic_query import assets_in_topic

        conn, cur = _mock_conn([])
        assets_in_topic(conn, topic_ko="요리", subtopic_ko="제빵")

        sql = _compact_sql(cur)
        params = cur.execute.call_args[0][1]
        self.assertIn("ge.topic->>'topic_ko' = %s", sql)
        self.assertIn("ge.topic->>'subtopic_ko' = %s", sql)
        # 값은 전부 %s 바인딩(인젝션 0)
        self.assertIn("요리", params)
        self.assertIn("제빵", params)

    def test_no_subtopic_filter_omits_predicate(self) -> None:
        from src.relations.topic_query import assets_in_topic

        conn, cur = _mock_conn([])
        assets_in_topic(conn, topic_ko="요리")

        sql = _compact_sql(cur)
        self.assertIn("ge.topic->>'topic_ko' = %s", sql)
        self.assertNotIn("subtopic_ko", sql)

    def test_sql_has_medical_exclusion(self) -> None:
        from src.relations.topic_query import assets_in_topic

        conn, cur = _mock_conn([])
        assets_in_topic(conn, topic_ko="요리")

        sql = _compact_sql(cur)
        self.assertIn("sa.domain_label IS DISTINCT FROM 'medical'", sql)
        self.assertIn("da.domain_label IS DISTINCT FROM 'medical'", sql)
        self.assertIn("ge.status = 'active'", sql)


if __name__ == "__main__":
    unittest.main()
