"""021 G1 — OpenSearch 검색 쿼리 빌더·결과 매핑 순수 함수 단위 테스트.

OS·DB·opensearch-py 불필요(순수). ``build_search_body``(하이브리드 쿼리 본문)와
``os_hit_to_row``(결과 행 매핑)가 020 인덱스 매핑(opensearch_sync.build_index_body)과
일치하고 media_search 버킷 행과 동형(SC-005)인지 충실히 검증한다 — 테스트 위조·약화 금지
(docs/테스트_가이드.md). 실제 OS 검색 실효는 G5(실OS e2e).
"""

from __future__ import annotations

import unittest

from src.search.opensearch_search import build_search_body, os_hit_to_row

# 020 인덱스의 nori 텍스트 필드(BM25 multi_match 대상). 필드명 정본 = opensearch_sync.build_index_body.
_NORI_TEXT_FIELDS = {"summary", "keywords", "labels", "file_name", "search_text"}


def _subqueries(body: dict) -> list:
    """hybrid 쿼리의 서브쿼리 목록을 꺼낸다."""
    return body["query"]["hybrid"]["queries"]


def _find_with_clause(subqueries: list, clause_key: str):
    """bool.must 에 주어진 절(clause_key)을 가진 서브쿼리·절을 찾는다(없으면 (None, None))."""
    for sub in subqueries:
        for clause in sub.get("bool", {}).get("must", []):
            if clause_key in clause:
                return sub, clause
    return None, None


class BuildSearchBodyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.query = "한국어 검색 질의"
        self.vector = [0.1, 0.2, 0.3, 0.4]
        self.body = build_search_body(self.query, self.vector, modality="text", k=50)

    def test_hybrid_with_two_subqueries(self) -> None:
        # (1) OpenSearch hybrid 쿼리 + 서브쿼리 2개(텍스트·knn).
        subs = _subqueries(self.body)
        self.assertEqual(len(subs), 2)

    def test_text_subquery_multi_match_nori_fields(self) -> None:
        # (1-①) nori 텍스트 multi_match — query 를 020 nori 필드 5개 대상으로.
        _, clause = _find_with_clause(_subqueries(self.body), "multi_match")
        self.assertIsNotNone(clause, "nori multi_match 서브쿼리가 있어야 한다")
        mm = clause["multi_match"]
        self.assertEqual(mm["query"], self.query)
        self.assertEqual(set(mm["fields"]), _NORI_TEXT_FIELDS)

    def test_knn_subquery_embedding_vector_k(self) -> None:
        # (1-②) knn 서브쿼리 — embedding 필드·query_vector·k.
        _, clause = _find_with_clause(_subqueries(self.body), "knn")
        self.assertIsNotNone(clause, "knn 서브쿼리가 있어야 한다")
        knn = clause["knn"]["embedding"]
        self.assertEqual(knn["vector"], self.vector)
        self.assertEqual(knn["k"], 50)

    def test_modality_filter_in_each_subquery(self) -> None:
        # (2) modality keyword term 필터가 각 서브쿼리 bool filter 에 포함.
        for sub in _subqueries(self.body):
            self.assertIn({"term": {"modality": "text"}}, sub["bool"]["filter"])

    def test_exclude_medical_must_not_default(self) -> None:
        # (3) exclude_medical=True(기본): domain_label='medical' must_not(FR-011).
        for sub in _subqueries(self.body):
            self.assertIn(
                {"term": {"domain_label": "medical"}}, sub["bool"]["must_not"]
            )

    def test_include_medical_when_disabled(self) -> None:
        # (3) exclude_medical=False: 의료 배제 미포함.
        body = build_search_body(
            self.query, self.vector, modality="text", exclude_medical=False
        )
        for sub in _subqueries(body):
            self.assertNotIn(
                {"term": {"domain_label": "medical"}},
                sub["bool"].get("must_not", []),
            )

    def test_deterministic_tiebreaker_sort(self) -> None:
        # (4) 결정적 tiebreaker(FR-009): 점수 desc → 동점 asset_id asc.
        self.assertEqual(
            self.body["sort"],
            [{"_score": {"order": "desc"}}, {"asset_id": {"order": "asc"}}],
        )

    def test_size_equals_k(self) -> None:
        # (5) size=k.
        self.assertEqual(self.body["size"], 50)

    def test_audio_modality_filter(self) -> None:
        # 모달리티 분담(text·audio→OS): audio 도 term 필터가 그 값으로 들어간다.
        body = build_search_body(self.query, self.vector, modality="audio")
        for sub in _subqueries(body):
            self.assertIn({"term": {"modality": "audio"}}, sub["bool"]["filter"])


class OsHitToRowTest(unittest.TestCase):
    def test_maps_to_media_search_bucket_shape(self) -> None:
        # OS hit → media_search 버킷 행 핵심 키(id·modality·similarity·summary·file_uri).
        hit = {
            "_id": "asset-uuid-1",
            "_score": 0.87,
            "_source": {
                "asset_id": "asset-uuid-1",
                "modality": "text",
                "domain_label": "general",
                "file_name": "보고서.txt",
                "fs_uri": "/data/보고서.txt",
                "summary": "요약 텍스트",
            },
        }
        row = os_hit_to_row(hit)
        self.assertEqual(row["id"], "asset-uuid-1")
        self.assertIsInstance(row["id"], str)
        self.assertEqual(row["modality"], "text")
        # similarity = _score(검색 파이프라인 정규화·융합 점수).
        self.assertEqual(row["similarity"], 0.87)
        self.assertEqual(row["summary"], "요약 텍스트")
        self.assertEqual(row["file_uri"], "/data/보고서.txt")

    def test_row_keys_homogeneous_with_media_search_bucket(self) -> None:
        # SC-005 응답 동형: media_search 버킷 행 핵심 키 집합과 동일(id·file_uri·modality·summary·similarity).
        hit = {
            "_id": "x",
            "_score": 1.0,
            "_source": {"asset_id": "x", "modality": "audio"},
        }
        row = os_hit_to_row(hit)
        self.assertEqual(
            set(row), {"id", "file_uri", "modality", "summary", "similarity"}
        )

    def test_missing_source_safe(self) -> None:
        # 엣지: _source 누락·_score None 안전 처리.
        row = os_hit_to_row({"_id": "y", "_score": None})
        self.assertEqual(row["id"], "y")
        self.assertEqual(row["similarity"], 0.0)
        self.assertEqual(row["summary"], "")
        self.assertEqual(row["file_uri"], "")
        self.assertIsNone(row["modality"])

    def test_none_meta_safe(self) -> None:
        # 엣지: 메타 필드 None 안전 처리(빈 문자열/None).
        hit = {
            "_id": "z",
            "_score": 0.5,
            "_source": {
                "asset_id": "z",
                "modality": None,
                "summary": None,
                "fs_uri": None,
            },
        }
        row = os_hit_to_row(hit)
        self.assertEqual(row["summary"], "")
        self.assertEqual(row["file_uri"], "")
        self.assertIsNone(row["modality"])

    def test_score_nan_safe(self) -> None:
        # 비유한 점수(NaN/inf) 방어 → 0.0.
        row = os_hit_to_row({"_id": "q", "_score": float("nan"), "_source": {"asset_id": "q"}})
        self.assertEqual(row["similarity"], 0.0)

    def test_asset_id_falls_back_to_underscore_id(self) -> None:
        # _source.asset_id 부재 시 _id 로 폴백(OS 색인이 _id=asset_id 라 동일).
        row = os_hit_to_row({"_id": "fallback-id", "_score": 0.3, "_source": {"modality": "text"}})
        self.assertEqual(row["id"], "fallback-id")


if __name__ == "__main__":
    unittest.main()
