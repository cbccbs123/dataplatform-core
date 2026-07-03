"""056 G7 — 포털 주제 표면 단위 테스트 (FR-501/502/503-facet/505).

전략(test_portal_api.py 패턴 재사용)
    FastAPI ``TestClient`` + auth bypass + ``_run_in_db`` passthrough. 주제 seam
    (``project_asset_topics``/``find_topic_neighbors``/``list_topics``/``assets_in_topic``)과
    검색 seam(``search_hybrid``)을 ``patch`` 로 대체해 **DB·OS·LLM·네트워크 없이** 순수 단위로 돈다.

검증 대상
    - 자산상세(``GET /assets/{id}``): 응답에 ``topics``(project_asset_topics) + ``same_topic_assets``
      (find_topic_neighbors·``already_linked`` 포함) 동반. 노출 게이트(None→404) 보존.
    - ``GET /topics`` → list_topics · ``GET /topics/{topic}`` → assets_in_topic 페이징(subtopic·limit·offset 전달).
    - ``GET /search`` → 응답 meta 에 주제 패싯 집계(``topic_facets``) · ``topic=``/``subtopic=`` 파라미터가
      parse_search_filters 를 거쳐 search_hybrid 의 ``search_filters`` 로 전달.
    - 전부 **신규 LLM 호출 0**(주제 seam·검색 seam 만).
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.app.portal_api import app

_AUTH_DISABLED_ENV = {
    "PORTAL_AUTH_DISABLED": "1",
    "PORTAL_JWT_SECRET": "test-secret",
}


def _passthrough_db(callback):
    """``_run_in_db`` 대역 — 가짜 conn 으로 callback 즉시 실행(DB 불필요·조회 함수는 patch)."""
    return callback(object())


def _empty_tiers(*_args, **_kwargs):
    return {}


def _enable_bypass(test_case: unittest.TestCase) -> None:
    env = patch.dict(os.environ, _AUTH_DISABLED_ENV, clear=False)
    env.start()
    test_case.addCleanup(env.stop)
    db = patch("src.app.portal_api._run_in_db", _passthrough_db)
    db.start()
    test_case.addCleanup(db.stop)


def _fake_search_result() -> dict:
    return {
        "query": "요리",
        "results": {
            "text_documents": [
                # 057-후속: 결과 행에 색인 topics 포함(패싯·클라 좁히기 소스 = 필터와 동일).
                {"id": "a1", "similarity": 0.9, "file_uri": "/x/a1.txt", "summary": "s1",
                 "topics": ["요리"], "subtopics": ["제빵"]},
                {"id": "a2", "similarity": 0.8, "file_uri": "/x/a2.txt", "summary": "s2",
                 "topics": ["요리"], "subtopics": []},
                {"id": "a3", "similarity": 0.7, "file_uri": "/x/a3.txt", "summary": "s3",
                 "topics": ["스포츠"], "subtopics": []},
            ],
        },
        "meta": {},
    }


class TestAssetDetailTopics(unittest.TestCase):
    """GET /assets/{id} — topics + same_topic_assets 보강(FR-501)."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.find_topic_neighbors")
    @patch("src.app.portal_api.project_asset_topics")
    @patch("src.app.portal_api.fetch_asset_detail")
    def test_detail_includes_topics_and_same_topic(
        self, mock_detail, mock_project, mock_neighbors
    ) -> None:
        mock_detail.return_value = {
            "asset_id": "a1", "modality": "text", "domain_label": "general",
            "status": "registered", "relations": [],
        }
        mock_project.return_value = [
            {"topic_ko": "요리", "subtopic_ko": "제빵", "topic_en": "cooking",
             "subtopic_en": "baking", "weight": 3},
        ]
        mock_neighbors.return_value = [
            {"asset_id": "a2", "shared_topics": ["요리"], "overlap_weight": 2,
             "already_linked": True},
            {"asset_id": "a7", "shared_topics": ["요리"], "overlap_weight": 1,
             "already_linked": False},
        ]
        resp = self.client.get("/assets/a1")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["topics"], mock_project.return_value)
        self.assertEqual(body["same_topic_assets"], mock_neighbors.return_value)
        # already_linked 표식 보존
        self.assertTrue(body["same_topic_assets"][0]["already_linked"])
        self.assertFalse(body["same_topic_assets"][1]["already_linked"])
        # seam 은 대상 자산으로 조회
        self.assertEqual(mock_project.call_args.kwargs["asset_id"], "a1")
        self.assertEqual(mock_neighbors.call_args.kwargs["asset_id"], "a1")

    @patch("src.app.portal_api.find_topic_neighbors")
    @patch("src.app.portal_api.project_asset_topics")
    @patch("src.app.portal_api.fetch_asset_detail")
    def test_detail_none_returns_404_no_topic_calls(
        self, mock_detail, mock_project, mock_neighbors
    ) -> None:
        # 노출 게이트: fetch_asset_detail None → 404, 주제 seam 미호출(불필요 조회 없음).
        mock_detail.return_value = None
        resp = self.client.get("/assets/nope")
        self.assertEqual(resp.status_code, 404)
        mock_project.assert_not_called()
        mock_neighbors.assert_not_called()


class TestTopicsList(unittest.TestCase):
    """GET /topics — list_topics 위임(FR-502)."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.list_topics")
    def test_list_topics(self, mock_list) -> None:
        mock_list.return_value = [
            {"topic_ko": "요리", "subtopic_ko": "제빵", "asset_count": 12},
            {"topic_ko": "스포츠", "subtopic_ko": None, "asset_count": 5},
        ]
        resp = self.client.get("/topics")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["topics"], mock_list.return_value)
        mock_list.assert_called_once()


class TestTopicAssets(unittest.TestCase):
    """GET /topics/{topic} — assets_in_topic 페이징(FR-502)."""

    def setUp(self) -> None:
        _enable_bypass(self)
        self.client = TestClient(app)

    @patch("src.app.portal_api.assets_in_topic")
    def test_topic_assets_paging(self, mock_assets) -> None:
        mock_assets.return_value = {
            "rows": [{"asset_id": "a1", "fs_uri": "/x/a1.txt", "file_name": "a1.txt"}],
            "total": 1,
        }
        resp = self.client.get(
            "/topics/요리", params={"subtopic": "제빵", "limit": 10, "offset": 5}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), mock_assets.return_value)
        kw = mock_assets.call_args.kwargs
        self.assertEqual(kw["topic_ko"], "요리")
        self.assertEqual(kw["subtopic_ko"], "제빵")
        self.assertEqual(kw["limit"], 10)
        self.assertEqual(kw["offset"], 5)

    @patch("src.app.portal_api.assets_in_topic")
    def test_topic_assets_no_subtopic(self, mock_assets) -> None:
        mock_assets.return_value = {"rows": [], "total": 0}
        resp = self.client.get("/topics/스포츠")
        self.assertEqual(resp.status_code, 200)
        self.assertIsNone(mock_assets.call_args.kwargs["subtopic_ko"])


class TestSearchTopicFacetAndFilter(unittest.TestCase):
    """GET /search — 주제 패싯 집계 + topic/subtopic 필터 전달(FR-503)."""

    def setUp(self) -> None:
        _enable_bypass(self)
        tiers = patch("src.app.portal_api.fetch_access_tiers", side_effect=_empty_tiers)
        tiers.start()
        self.addCleanup(tiers.stop)
        self.client = TestClient(app)

    @patch("src.app.portal_api.search_hybrid")
    def test_search_returns_topic_facet(self, mock_search) -> None:
        # 057-후속: 패싯은 결과 행의 **색인 topics**(=필터 소스)로 집계 — project_asset_topics 미사용
        # (라이브 투영 대비 소스 불일치·N+1 제거). 프론트는 이 행 topics 로 클라 좁히기.
        mock_search.return_value = _fake_search_result()
        resp = self.client.get("/search", params={"q": "요리", "size": 10})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        facets = body["meta"]["topic_facets"]
        # 요리={a1,a2} 2건(하위 제빵 a1 1건), 스포츠={a3} 1건(하위 없음). 결과-스코프 nested·결정적 정렬.
        self.assertEqual(
            facets,
            [
                {"topic_ko": "요리", "asset_count": 2,
                 "subtopics": [{"subtopic_ko": "제빵", "asset_count": 1}]},
                {"topic_ko": "스포츠", "asset_count": 1, "subtopics": []},
            ],
        )
        # 결과 행에도 topics 노출(프론트 클라 좁히기용) → 패싯 클릭 = 이 topics 로 필터.
        rows = [r for bucket in body["results"].values() for r in bucket]
        self.assertTrue(any(r.get("topics") == ["요리"] for r in rows))

    @patch("src.app.portal_api.project_asset_topics", return_value=[])
    @patch("src.app.portal_api.search_hybrid")
    def test_search_passes_topic_filter(self, mock_search, _mock_project) -> None:
        mock_search.return_value = _fake_search_result()
        resp = self.client.get(
            "/search", params={"q": "요리", "topic": "요리", "subtopic": "제빵"}
        )
        self.assertEqual(resp.status_code, 200)
        sf = mock_search.call_args.kwargs["search_filters"]
        self.assertIsNotNone(sf)
        self.assertEqual(sf.topic, "요리")
        self.assertEqual(sf.subtopic, "제빵")


if __name__ == "__main__":
    unittest.main()
