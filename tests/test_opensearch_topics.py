"""OpenSearch 주제 색인 단위 테스트 — DB·OS·opensearch-py 실서버 불필요 (056 G4).

검증 의도 (FR-201~204·SC-03)
    - ``build_index_body`` 매핑에 주제 2필드가 존재한다: ``topics``(keyword)·``subtopics``(keyword).
      검색 패싯/필터(keyword·terms)용이다. (topics_text BM25 보강 필드는 스코프 철회로 **색인하지 않는다**.)
    - ``asset_to_doc(row, ch, topics=[...])`` 는 주제를 문서 필드로 수록한다(순수·결정적):
      ``topics``=dedup 된 topic_ko, ``subtopics``=subtopic_ko.
      ``topics=None``(또는 생략)이면 두 필드를 **넣지 않는다**(관계 없는 자산·하위호환).
    - ``update_asset_topics(client,index,asset_id,topics)`` 는 OS ``update`` 부분문서
      (``body={"doc": {...}}``)로 주제 2필드만 갱신한다(전체 재색인 아님·G5 재색인 훅의 seam).
    - T403 배선: ``index_asset``/``sync_all`` 전체문서 색인 경로가 ``project_asset_topics`` 로
      **현재 active 주제를 항상 함께 실어**, 재수집/재색인이 topics 를 지우지 않게 한다(C5·SC-03).

모든 검증은 순수 함수 + 가짜 client/conn 주입으로 수행한다 — 실 OS·DB 색인은 T404(사람 게이트).
"""
from __future__ import annotations

import unittest
from unittest import mock

from src.search.opensearch_sync import (
    _topics_doc_fields,
    asset_to_doc,
    build_index_body,
)


def _topic(topic_ko, subtopic_ko, topic_en, subtopic_en, weight=1):
    return {
        "topic_ko": topic_ko,
        "subtopic_ko": subtopic_ko,
        "topic_en": topic_en,
        "subtopic_en": subtopic_en,
        "weight": weight,
    }


def _row(**over):
    row = {
        "asset_id": "a1",
        "modality": "video",
        "domain_label": "general",
        "status": "registered",
        "fs_path": "/data/무선_충전기.mp4",
        "ext_meta": {"summary": "요약", "keywords": ["k"], "labels": ["l"]},
        "emb": "[0.1,0.2,0.3]",
        "chunk_count": 3,
    }
    row.update(over)
    return row


class TestIndexBodyTopicMappings(unittest.TestCase):
    """T401 — build_index_body 에 주제 keyword 필드만 존재(topics_text 미색인)."""

    def test_topics_and_subtopics_are_keyword(self) -> None:
        props = build_index_body()["mappings"]["properties"]
        self.assertEqual(props["topics"]["type"], "keyword")
        self.assertEqual(props["subtopics"]["type"], "keyword")

    def test_topics_text_not_indexed(self) -> None:
        # topics_text(BM25 관련도 보강)는 스코프 철회 — 매핑에 없어야 한다(패싯/필터 keyword 만 색인).
        props = build_index_body()["mappings"]["properties"]
        self.assertNotIn("topics_text", props)

    def test_topic_pairs_is_keyword(self) -> None:
        # 059 FR-102 — 부모>자식 짝 보존 필드. topics/subtopics 옆에 keyword(패싯·정확필터).
        props = build_index_body()["mappings"]["properties"]
        self.assertEqual(props["topic_pairs"]["type"], "keyword")


class TestTopicsDocFieldsTopicPairs(unittest.TestCase):
    """T101 — _topics_doc_fields 의 topic_pairs 조립(순수·결정적) + 평면필드 불변."""

    def test_topic_pairs_join_parent_child(self) -> None:
        # 각 짝은 "topic_ko>subtopic_ko" — 부모·자식이 한 문자열로 보존된다(교차곱 오배치 방지).
        fields = _topics_doc_fields(
            [
                _topic("음식·요리", "먹방", "food", "mukbang"),
                _topic("IT·기술", "데이터", "it", "data"),
            ]
        )
        self.assertEqual(fields["topic_pairs"], ["음식·요리>먹방", "IT·기술>데이터"])

    def test_topic_pairs_standalone_when_no_subtopic(self) -> None:
        # subtopic 이 None/"" 이면 topic_ko 단독으로 넣는다(짝 없는 주제도 트리 루트로 표시).
        fields = _topics_doc_fields(
            [_topic("음식·요리", None, "food", ""), _topic("음악", "", "music", "")]
        )
        self.assertEqual(fields["topic_pairs"], ["음식·요리", "음악"])

    def test_topic_pairs_dedup_in_order(self) -> None:
        # 같은 짝 반복은 첫 등장만 남기고 입력 순서 보존(결정적, _dedup_in_order 재사용).
        fields = _topics_doc_fields(
            [
                _topic("음식·요리", "먹방", "food", "mukbang"),
                _topic("IT·기술", "데이터", "it", "data"),
                _topic("음식·요리", "먹방", "food", "mukbang"),  # 중복 짝
                _topic("음식·요리", None, "food", ""),  # 단독(다른 문자열)
            ]
        )
        self.assertEqual(
            fields["topic_pairs"],
            ["음식·요리>먹방", "IT·기술>데이터", "음식·요리"],
        )

    def test_spec_example_pairs_and_flat_fields(self) -> None:
        # spec 059 개요 예시: 멀티토픽 자산의 짝 보존 + 평면 필드 값(회귀 가드).
        fields = _topics_doc_fields(
            [
                _topic("음식·요리", "먹방", "food", "mukbang"),
                _topic("IT·기술", "데이터", "it", "data"),
                _topic("음식·요리", None, "food", ""),
            ]
        )
        self.assertEqual(
            fields["topic_pairs"], ["음식·요리>먹방", "IT·기술>데이터", "음식·요리"]
        )
        # 평면 topics/subtopics 는 059 전과 **바이트 동일**해야 한다(랭킹·필터 무회귀).
        self.assertEqual(fields["topics"], ["음식·요리", "IT·기술"])
        self.assertEqual(fields["subtopics"], ["먹방", "데이터"])

    def test_flat_fields_unchanged_by_topic_pairs_addition(self) -> None:
        # 회귀 가드: topic_pairs 추가가 기존 평면 topics/subtopics 로직을 건드리지 않는다.
        topics = [
            _topic("요리", "제빵", "cooking", "baking"),
            _topic("요리", "제과", "cooking", "confectionery"),
            _topic("음악", "재즈", "music", "jazz"),
        ]
        fields = _topics_doc_fields(topics)
        self.assertEqual(fields["topics"], ["요리", "음악"])
        self.assertEqual(fields["subtopics"], ["제빵", "제과", "재즈"])


class TestAssetToDocTopics(unittest.TestCase):
    """T401 — asset_to_doc 의 topics 수록·하위호환."""

    def test_topics_present_populates_two_fields(self) -> None:
        doc = asset_to_doc(
            _row(),
            channel="st",
            topics=[_topic("요리", "제빵", "cooking", "baking", weight=2)],
        )
        self.assertEqual(doc["topics"], ["요리"])
        self.assertEqual(doc["subtopics"], ["제빵"])
        # 059 FR-101 — asset_to_doc(전체문서)도 단일 출처(_topics_doc_fields) 경유라 짝 자동 반영.
        self.assertEqual(doc["topic_pairs"], ["요리>제빵"])
        # topics_text(BM25 보강)는 색인하지 않는다 — keyword 패싯/필터만.
        self.assertNotIn("topics_text", doc)

    def test_topic_pairs_present_in_full_doc(self) -> None:
        # 멀티토픽(짝 손실 케이스) — 전체문서에 짝이 부모별로 보존된다(교차곱 방지·SC-02 계약).
        doc = asset_to_doc(
            _row(),
            channel="st",
            topics=[
                _topic("음식·요리", "먹방", "food", "mukbang"),
                _topic("IT·기술", "데이터", "it", "data"),
            ],
        )
        self.assertEqual(doc["topic_pairs"], ["음식·요리>먹방", "IT·기술>데이터"])

    def test_multiple_topics_dedup_topic_ko_in_order(self) -> None:
        # 같은 topic_ko 가 여러 subtopic 으로 오면 topics(keyword)는 dedup, 입력 순서 보존(결정적).
        doc = asset_to_doc(
            _row(),
            channel="st",
            topics=[
                _topic("요리", "제빵", "cooking", "baking"),
                _topic("요리", "제과", "cooking", "confectionery"),
                _topic("음악", "재즈", "music", "jazz"),
            ],
        )
        self.assertEqual(doc["topics"], ["요리", "음악"])
        self.assertEqual(doc["subtopics"], ["제빵", "제과", "재즈"])

    def test_none_topics_omits_both_fields(self) -> None:
        # 관계 없는 자산(topics=None) → 두 필드 부재(하위호환·기존 문서 형상 불변).
        doc = asset_to_doc(_row(), channel="st")
        self.assertNotIn("topics", doc)
        self.assertNotIn("subtopics", doc)

    def test_empty_topics_list_omits_both_fields(self) -> None:
        # 빈 리스트(투영 결과 주제 0)도 두 필드 부재 — None 과 동일(관계 없는 자산).
        doc = asset_to_doc(_row(), channel="st", topics=[])
        self.assertNotIn("topics", doc)
        self.assertNotIn("subtopics", doc)

    def test_none_or_empty_subtopic_skipped_in_subtopics(self) -> None:
        # subtopic_ko 가 None/"" 이면 subtopics(keyword)에서 제외(빈 keyword 리스트).
        doc = asset_to_doc(
            _row(),
            channel="st",
            topics=[_topic("요리", None, "cooking", ""), _topic("음악", "", "music", "jazz")],
        )
        self.assertEqual(doc["topics"], ["요리", "음악"])
        self.assertEqual(doc["subtopics"], [])  # None/"" subtopic 스킵 → 빈 keyword 리스트

    def test_topics_do_not_disturb_existing_fields(self) -> None:
        # 주제 수록이 기존 문서 필드(summary·embedding 등)를 훼손하지 않는다.
        doc = asset_to_doc(_row(), channel="st", topics=[_topic("요리", "제빵", "cooking", "baking")])
        self.assertEqual(doc["asset_id"], "a1")
        self.assertEqual(doc["summary"], "요약")
        self.assertEqual(doc["embedding"], [0.1, 0.2, 0.3])


class TestUpdateAssetTopics(unittest.TestCase):
    """T401 — update_asset_topics: OS 부분 update(body={"doc": {...}})."""

    def test_partial_update_body_shape(self) -> None:
        from src.search.opensearch_sync import update_asset_topics

        client = mock.MagicMock()
        update_asset_topics(
            client,
            "assets",
            "a1",
            [_topic("요리", "제빵", "cooking", "baking", weight=2)],
        )
        client.update.assert_called_once()
        _args, kwargs = client.update.call_args
        # index_asset 의 client.index(index=, id=, body=) 패턴을 미러 — update 도 동형.
        self.assertEqual(kwargs["index"], "assets")
        self.assertEqual(kwargs["id"], "a1")
        body = kwargs["body"]
        self.assertIn("doc", body)  # 부분 문서 갱신(전체 재색인 아님)
        partial = body["doc"]
        self.assertEqual(partial["topics"], ["요리"])
        self.assertEqual(partial["subtopics"], ["제빵"])
        # 059 FR-101 — 부분문서 갱신도 단일 출처(_topics_doc_fields) 경유라 topic_pairs 자동 반영.
        self.assertEqual(partial["topic_pairs"], ["요리>제빵"])
        self.assertNotIn("topics_text", partial)  # BM25 보강 필드 미색인

    def test_id_coerced_to_str(self) -> None:
        # asset_id 가 비-str(UUID 등)이어도 OS _id 는 항상 str(index_asset 관례 미러).
        import uuid

        from src.search.opensearch_sync import update_asset_topics

        client = mock.MagicMock()
        u = uuid.uuid4()
        update_asset_topics(client, "assets", u, [_topic("요리", "제빵", "cooking", "baking")])
        self.assertEqual(client.update.call_args.kwargs["id"], str(u))

    def test_empty_topics_clears_fields(self) -> None:
        # 관계 강등/제거(SC-02) 시 topics 를 **빈 값으로 갱신**해 stale 주제를 지운다.
        from src.search.opensearch_sync import update_asset_topics

        client = mock.MagicMock()
        update_asset_topics(client, "assets", "a1", [])
        partial = client.update.call_args.kwargs["body"]["doc"]
        self.assertEqual(partial["topics"], [])
        self.assertEqual(partial["subtopics"], [])


# ── T403 배선 테스트용 가짜 client/conn (실 OS·DB 없이 색인 액션 조립 검증) ──


class _FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)
        self.executed: list[tuple[str, object]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def execute(self, sql: str, params=None) -> None:
        self.executed.append((sql, params))

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class _FakeConn:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = list(rows)

    def cursor(self, row_factory=None) -> _FakeCursor:
        return _FakeCursor(self._rows)


class _FakeIndices:
    def __init__(self) -> None:
        self._exists = False

    def exists(self, index: str) -> bool:
        return self._exists

    def create(self, index: str, body: dict) -> None:
        self._exists = True

    def delete(self, index: str) -> None:
        self._exists = False

    def refresh(self, index: str) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.indexed: list[dict] = []
        self.bulk_calls: list[list] = []
        self.indices = _FakeIndices()

    def index(self, *, index: str, id: str, body: dict) -> dict:
        self.indexed.append({"index": index, "id": id, "body": body})
        return {"result": "updated"}

    def bulk(self, actions, **kwargs) -> tuple[int, list]:
        acts = list(actions)
        self.bulk_calls.append(acts)
        return (len(acts), [])


_PROJECT = "src.relations.topic_query.project_asset_topics"


class TestIndexAssetTopicWiring(unittest.TestCase):
    """T403 — 전체문서 색인 경로(index_asset)가 현재 active 주제를 항상 실어야 한다(C5·SC-03).

    ``index_asset`` 는 기본 topics_fn 으로 ``project_asset_topics(conn, asset_id)`` 를 계산해
    ``asset_to_doc(..., topics=...)`` 에 주입한다. 관계가 있으면 재색인 문서가 topics 를 포함하므로
    재수집/재색인이 이미 색인된 topics 를 지우지 않는다(self-heal).
    """

    @mock.patch(_PROJECT)
    def test_index_asset_default_path_includes_current_topics(self, m_project) -> None:
        from src.search.opensearch_sync import index_asset

        m_project.return_value = [_topic("요리", "제빵", "cooking", "baking", weight=2)]
        client = _FakeClient()
        conn = _FakeConn([_row()])

        doc = index_asset(client, conn, "a1", index="assets", channel="st")

        # 색인 문서가 현재 active 주제를 포함(C5) — topics_fn 미주입 시 project_asset_topics 계산.
        self.assertEqual(doc["topics"], ["요리"])
        self.assertEqual(doc["subtopics"], ["제빵"])
        self.assertEqual(client.indexed[0]["body"]["topics"], ["요리"])
        # project 는 conn + 해당 asset_id 로 호출(투영 seam 재사용)
        m_project.assert_called_once()
        self.assertEqual(m_project.call_args.kwargs.get("asset_id"), "a1")

    @mock.patch(_PROJECT)
    def test_index_asset_no_relations_omits_topics(self, m_project) -> None:
        # 관계 없는 자산 → 투영 [] → 문서에 topics 필드 부재(하위호환·기존 문서 형상 유지).
        from src.search.opensearch_sync import index_asset

        m_project.return_value = []
        client = _FakeClient()
        doc = index_asset(client, _FakeConn([_row()]), "a1", index="assets", channel="st")
        self.assertNotIn("topics", doc)
        self.assertNotIn("subtopics", doc)

    def test_index_asset_topics_fn_injection_overrides_default(self) -> None:
        # topics_fn 주입 seam(테스트/특수 경로) — 주입 함수 결과가 문서에 실린다.
        from src.search.opensearch_sync import index_asset

        client = _FakeClient()
        doc = index_asset(
            client,
            _FakeConn([_row()]),
            "a1",
            index="assets",
            channel="st",
            topics_fn=lambda conn, aid: [_topic("음악", "재즈", "music", "jazz")],
        )
        self.assertEqual(doc["topics"], ["음악"])

    @mock.patch(_PROJECT)
    def test_sync_all_bulk_docs_include_topics(self, m_project) -> None:
        # 전체 재색인(백필·T801) 경로도 현재 주제를 실어야 한다 — full reindex self-heal(R3).
        from src.search.opensearch_sync import sync_all

        m_project.return_value = [_topic("요리", "제빵", "cooking", "baking")]
        client = _FakeClient()
        conn = _FakeConn([_row(asset_id="a1")])

        def _bulk(cl, actions, **kw):
            return cl.bulk(actions, **kw)

        sync_all(client, conn, index="assets", channel="st", bulk_fn=_bulk)
        action = client.bulk_calls[0][0]
        self.assertEqual(action["_source"]["topics"], ["요리"])


if __name__ == "__main__":
    unittest.main()
