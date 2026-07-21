"""065 자기주제 정본 소비 함수 단위 테스트 (mock, DB·LLM 불필요) — 주제 패싯·브라우즈(FR-402).

검증 의도 (FR-402·FR-403)
    포탈 주제 패싯(``list_topics``·구 060)·주제별 자산(``assets_in_topic``·같은-주제 탐색)을 이웃-엣지
    투영이 아니라 **자기주제 정본(``asset_topic``) 조인**으로 계산한다 — 응답 계약(필드명·정렬)은 구
    ``src.relations.topic_query`` 함수와 동일해 포탈 프론트 무변경 스왑이 가능하다. 구 경로가 읽던
    ``graph_edge.topic``(관계 맥락 라벨) 소비를 끊어 주제 소스를 하나(정본)로 단일화한다(FR-403).

    구 ``tests/test_topic_query.py`` 의 list_topics/assets_in_topic 시나리오를 정본 기준으로 이식한다.
    mock conn 패턴은 ``tests/test_asset_topic_classify.py`` 동형(cursor 컨텍스트·fetchall 주입).
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock


def _mock_conn(fetchall_val):
    """``conn.cursor(row_factory=dict_row)`` 컨텍스트매니저 mock — fetchall 을 주입값으로 통제."""
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchall.return_value = fetchall_val
    conn.cursor.return_value = cur
    return conn, cur


class TestListTopics(unittest.TestCase):
    """``list_topics`` — (topic_ko, subtopic_ko) 별 distinct 자산 수 + 주제 전체 수(정본 조인)."""

    def test_groups_by_pair_with_asset_and_topic_counts(self) -> None:
        from src.topic.asset_topic_query import list_topics

        rows = [
            {"topic_ko": "요리", "subtopic_ko": "제빵", "asset_id": "a1"},
            {"topic_ko": "요리", "subtopic_ko": "제빵", "asset_id": "a2"},
            {"topic_ko": "요리", "subtopic_ko": None, "asset_id": "a3"},
            {"topic_ko": "스포츠", "subtopic_ko": None, "asset_id": "a4"},
        ]
        conn, _ = _mock_conn(rows)
        out = list_topics(conn)
        # 정렬: topic_ko asc → subtopic_ko asc(None="" 최상단). '스'<'요'.
        self.assertEqual(
            out,
            [
                {"topic_ko": "스포츠", "subtopic_ko": None, "asset_count": 1,
                 "topic_asset_count": 1},
                {"topic_ko": "요리", "subtopic_ko": None, "asset_count": 1,
                 "topic_asset_count": 3},
                {"topic_ko": "요리", "subtopic_ko": "제빵", "asset_count": 2,
                 "topic_asset_count": 3},
            ],
        )

    def test_empty_topic_ko_skipped_and_subtopic_normalized(self) -> None:
        from src.topic.asset_topic_query import list_topics

        rows = [
            {"topic_ko": "", "subtopic_ko": "무시", "asset_id": "x"},   # 빈 topic → 제외
            {"topic_ko": "음악", "subtopic_ko": "", "asset_id": "m1"},  # "" → None 정규화
        ]
        conn, _ = _mock_conn(rows)
        out = list_topics(conn)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["topic_ko"], "음악")
        self.assertIsNone(out[0]["subtopic_ko"])

    def test_sql_excludes_medical(self) -> None:
        from src.topic.asset_topic_query import list_topics

        conn, cur = _mock_conn([])
        list_topics(conn)
        sql = " ".join(cur.execute.call_args[0][0].split()).lower()
        self.assertIn("from asset_topic", sql)
        self.assertIn("domain_label is distinct from 'medical'", sql)

    def test_empty_rows_returns_empty(self) -> None:
        from src.topic.asset_topic_query import list_topics

        conn, _ = _mock_conn([])
        self.assertEqual(list_topics(conn), [])

    def test_multiple_topics_deterministic_order(self) -> None:
        from src.topic.asset_topic_query import list_topics

        rows = [
            {"topic_ko": "여행", "subtopic_ko": "국내", "asset_id": "t1"},
            {"topic_ko": "게임", "subtopic_ko": None, "asset_id": "g1"},
            {"topic_ko": "게임", "subtopic_ko": "RPG", "asset_id": "g2"},
        ]
        conn, _ = _mock_conn(rows)
        out = list_topics(conn)
        # topic_ko asc('게임'<'여행') · 같은 topic 내 subtopic None 먼저.
        self.assertEqual(
            [(o["topic_ko"], o["subtopic_ko"]) for o in out],
            [("게임", None), ("게임", "RPG"), ("여행", "국내")],
        )
        # 재실행 동일(결정적·헌법 3조).
        conn2, _ = _mock_conn(rows)
        self.assertEqual(list_topics(conn2), out)

    def test_topic_asset_count_uniform_across_subtopics(self) -> None:
        from src.topic.asset_topic_query import list_topics

        rows = [
            {"topic_ko": "요리", "subtopic_ko": "제빵", "asset_id": "a1"},
            {"topic_ko": "요리", "subtopic_ko": "한식", "asset_id": "a2"},
        ]
        conn, _ = _mock_conn(rows)
        out = list_topics(conn)
        # 같은 topic 의 모든 행은 동일한 topic_asset_count(주제 전체 distinct) — 하위주제 합산 아님.
        self.assertEqual({o["topic_asset_count"] for o in out}, {2})
        self.assertEqual({o["subtopic_ko"] for o in out}, {"제빵", "한식"})


class TestAssetsInTopic(unittest.TestCase):
    """``assets_in_topic`` — 특정 주제(정본) 자산 페이징(asset_id asc·file_name basename)."""

    def _rows(self):
        return [
            {"asset_id": "a3", "fs_uri": "/x/a3.txt", "fs_path": "/data/a3__요리.txt",
             "modality": "text", "keywords": ["요리", "레시피"], "labels": None},
            {"asset_id": "a1", "fs_uri": "/x/a1.txt", "fs_path": "/data/a1__빵.jpg",
             "modality": "image", "keywords": ["빵"], "labels": [{"label": "bread", "score": 0.9}]},
            {"asset_id": "a2", "fs_uri": "/x/a2.txt", "fs_path": "/data/a2__국.mp4",
             "modality": "video", "keywords": None, "labels": None},
        ]

    def test_shape_sorted_and_basename(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, _ = _mock_conn(self._rows())
        out = assets_in_topic(conn, topic_ko="요리")
        self.assertEqual(out["total"], 3)
        self.assertEqual([r["asset_id"] for r in out["rows"]], ["a1", "a2", "a3"])  # asc
        r0 = out["rows"][0]  # a1
        self.assertEqual(set(r0.keys()),
                         {"asset_id", "fs_uri", "file_name", "modality", "keywords", "labels"})
        self.assertEqual(r0["file_name"], "a1__빵.jpg")  # fs_path basename
        self.assertIsInstance(r0["asset_id"], str)
        self.assertEqual(r0["modality"], "image")
        self.assertEqual(r0["keywords"], ["빵"])
        self.assertEqual(r0["labels"], ["bread"])   # [{label,score}] → label 만
        # modality_counts = 필터 전 전체 분포(모달리티 폴더 카운트)
        self.assertEqual(out["modality_counts"], {"text": 1, "image": 1, "video": 1})

    def test_modality_filter_counts_stay_full(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, _ = _mock_conn(self._rows())
        out = assets_in_topic(conn, topic_ko="요리", modality="image")
        self.assertEqual(out["total"], 1)                     # rows 는 image 만
        self.assertEqual([r["asset_id"] for r in out["rows"]], ["a1"])
        # counts 는 modality 필터와 무관하게 전체(폴더 카운트가 클릭으로 줄면 안 됨)
        self.assertEqual(out["modality_counts"], {"text": 1, "image": 1, "video": 1})

    def test_paging(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, _ = _mock_conn(self._rows())
        page = assets_in_topic(conn, topic_ko="요리", limit=2, offset=1)
        self.assertEqual(page["total"], 3)  # total 은 페이징 전 실수
        self.assertEqual([r["asset_id"] for r in page["rows"]], ["a2", "a3"])

    def test_subtopic_filter_appended(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, cur = _mock_conn([])
        assets_in_topic(conn, topic_ko="요리", subtopic_ko="제빵")
        sql = " ".join(cur.execute.call_args[0][0].split())
        self.assertIn("at.subtopic_ko = %s", sql)
        self.assertIn("제빵", cur.execute.call_args[0][1])

    def test_no_subtopic_no_filter(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, cur = _mock_conn([])
        assets_in_topic(conn, topic_ko="스포츠")
        sql = " ".join(cur.execute.call_args[0][0].split())
        self.assertNotIn("at.subtopic_ko = %s", sql)

    def test_unassigned_only_filters_null(self) -> None:
        """unassigned_only=True 면 '기타'(subtopic 미부여)만 — subtopic_ko IS NULL(값 매칭 아님)."""
        from src.topic.asset_topic_query import assets_in_topic

        conn, cur = _mock_conn([])
        assets_in_topic(conn, topic_ko="요리", unassigned_only=True)
        sql = " ".join(cur.execute.call_args[0][0].split())
        self.assertIn("at.subtopic_ko IS NULL", sql)
        self.assertNotIn("at.subtopic_ko = %s", sql)  # 값 바인딩이 아니라 NULL 필터
        self.assertEqual(list(cur.execute.call_args[0][1]), ["요리"])  # subtopic 값 파라미터 없음

    def test_unassigned_only_overrides_subtopic(self) -> None:
        """unassigned_only=True 는 subtopic_ko 지정보다 우선(둘 다 오면 IS NULL)."""
        from src.topic.asset_topic_query import assets_in_topic

        conn, cur = _mock_conn([])
        assets_in_topic(conn, topic_ko="요리", subtopic_ko="제빵", unassigned_only=True)
        sql = " ".join(cur.execute.call_args[0][0].split())
        self.assertIn("at.subtopic_ko IS NULL", sql)
        self.assertNotIn("at.subtopic_ko = %s", sql)

    def test_sql_excludes_medical(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, cur = _mock_conn([])
        assets_in_topic(conn, topic_ko="요리")
        sql = " ".join(cur.execute.call_args[0][0].split()).lower()
        self.assertIn("from asset_topic", sql)
        self.assertIn("domain_label is distinct from 'medical'", sql)

    def test_empty_result(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, _ = _mock_conn([])
        self.assertEqual(assets_in_topic(conn, topic_ko="없음"),
                         {"rows": [], "total": 0, "modality_counts": {}})

    def test_offset_beyond_total(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, _ = _mock_conn(self._rows())
        out = assets_in_topic(conn, topic_ko="요리", offset=10)
        self.assertEqual(out["total"], 3)   # total 은 페이징 전 실수(불변)
        self.assertEqual(out["rows"], [])   # offset 이 total 초과 → 빈 페이지

    def test_fs_uri_preserved(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, _ = _mock_conn(self._rows())
        out = assets_in_topic(conn, topic_ko="요리")
        # fs_uri 원본 컬럼 보존(파일명은 fs_path basename 파생).
        self.assertEqual(
            {r["fs_uri"] for r in out["rows"]},
            {"/x/a1.txt", "/x/a2.txt", "/x/a3.txt"},
        )

    def test_null_fs_path_safe_basename(self) -> None:
        from src.topic.asset_topic_query import assets_in_topic

        conn, _ = _mock_conn([{"asset_id": "z1", "fs_uri": None, "fs_path": None, "modality": "text"}])
        out = assets_in_topic(conn, topic_ko="요리")
        self.assertEqual(out["total"], 1)
        self.assertEqual(out["rows"][0]["file_name"], "")  # None fs_path → 빈 basename(안전)
        self.assertIsNone(out["rows"][0]["fs_uri"])


class TestAssetsUnclassified(unittest.TestCase):
    """assets_unclassified — 주제 미부여 자산(파일탐색기 '미분류' 폴더·전수 포함)."""

    def test_sql_left_join_null_registered_nonmedical(self) -> None:
        from src.topic.asset_topic_query import assets_unclassified

        conn, cur = _mock_conn([])
        assets_unclassified(conn, limit=50, offset=0)
        sql = " ".join(cur.execute.call_args[0][0].split()).lower()
        self.assertIn("left join asset_topic", sql)          # 미부여 회수(부여된 것 제외)
        self.assertIn("at.asset_id is null", sql)            # 주제 정본 없음만
        self.assertIn("a.status = 'registered'", sql)        # 수집 중/실패 제외
        self.assertIn("domain_label is distinct from 'medical'", sql)  # PHI 제외 상속

    def test_shape_sorted_with_modality(self) -> None:
        from src.topic.asset_topic_query import assets_unclassified

        conn, _ = _mock_conn([
            {"asset_id": "a2", "fs_uri": "/x/a2", "fs_path": "/d/a2__x.mp3", "modality": "audio"},
            {"asset_id": "a1", "fs_uri": "/x/a1", "fs_path": "/d/a1__y.jpg", "modality": "image"},
        ])
        out = assets_unclassified(conn)
        self.assertEqual(out["total"], 2)
        self.assertEqual([r["asset_id"] for r in out["rows"]], ["a1", "a2"])  # asset_id asc
        self.assertEqual(set(out["rows"][0].keys()),
                         {"asset_id", "fs_uri", "file_name", "modality", "keywords", "labels"})
        self.assertEqual(out["rows"][0]["modality"], "image")  # 파일 아이콘용 modality 포함
        self.assertEqual(out["modality_counts"], {"audio": 1, "image": 1})  # 모달리티 폴더 카운트


if __name__ == "__main__":
    unittest.main()
