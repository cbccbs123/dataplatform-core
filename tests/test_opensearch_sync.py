"""OpenSearch 동기화 순수 함수 단위 테스트 — DB·OS·opensearch-py 불필요.

문서 빌더·벡터 파싱·인덱스 매핑만 검증한다(실제 색인 IO 는 G2 이후 실DB/실OS e2e 책임).
순수 함수(`build_index_body`·`asset_to_doc`·`parse_vector`)는 결정적이고 입력만으로
출력이 정해지므로 CI 단위 게이트에서 항상 돈다.
"""
from __future__ import annotations

import unittest

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.search.opensearch_sync import asset_to_doc, build_index_body, parse_vector


class TestParseVector(unittest.TestCase):
    def test_list_passthrough(self) -> None:
        # 이미 리스트면 float 로 정규화해 통과.
        self.assertEqual(parse_vector([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3])

    def test_pgvector_string(self) -> None:
        # pgvector 텍스트 표현 '[..]' 를 float 리스트로 파싱(음수 포함).
        self.assertEqual(parse_vector("[0.5,-0.25,1.0]"), [0.5, -0.25, 1.0])

    def test_empty_and_whitespace(self) -> None:
        # 빈 벡터·공백 토큰을 안전하게 처리.
        self.assertEqual(parse_vector("[]"), [])
        self.assertEqual(parse_vector("[ 0.1 , 0.2 ]"), [0.1, 0.2])


class TestAssetToDoc(unittest.TestCase):
    def _row(self, **over):
        row = {
            "asset_id": "a1",
            "modality": "video",
            "domain_label": "general",
            "status": "registered",
            "fs_path": "/data/sub/무선_충전기_xyz.mp4",
            "ext_meta": {
                "summary": "무선 충전기 리뷰",
                "keywords": ["충전기", "Qi2"],
                "labels": ["전자제품"],
            },
            "emb": "[0.1,0.2,0.3]",
            "chunk_count": 7,
        }
        row.update(over)
        return row

    def test_doc_shape_and_fields(self) -> None:
        doc = asset_to_doc(self._row(), channel="st")
        self.assertEqual(doc["asset_id"], "a1")
        self.assertEqual(doc["modality"], "video")
        self.assertEqual(doc["domain_label"], "general")
        self.assertEqual(doc["status"], "registered")
        self.assertEqual(doc["channel"], "st")
        self.assertEqual(doc["file_name"], "무선_충전기_xyz.mp4")
        self.assertEqual(doc["fs_uri"], "/data/sub/무선_충전기_xyz.mp4")
        self.assertEqual(doc["summary"], "무선 충전기 리뷰")
        self.assertEqual(doc["keywords"], ["충전기", "Qi2"])
        self.assertEqual(doc["labels"], ["전자제품"])
        self.assertEqual(doc["chunk_count"], 7)
        self.assertEqual(doc["embedding"], [0.1, 0.2, 0.3])

    def test_all_expected_keys_present(self) -> None:
        # T001 이 명시한 문서 필드 집합을 전부 갖는지(누락 방지).
        doc = asset_to_doc(self._row(), channel="st")
        expected = {
            "asset_id", "modality", "domain_label", "status", "channel",
            "file_name", "fs_uri", "summary", "keywords", "labels",
            "search_text", "chunk_count", "embedding",
        }
        self.assertTrue(expected.issubset(doc.keys()))

    def test_search_text_concatenates(self) -> None:
        # BM25 대상 search_text 가 summary·file_name·keywords·labels 를 모두 포함.
        st = asset_to_doc(self._row(), channel="st")["search_text"]
        for token in ("무선 충전기 리뷰", "무선_충전기_xyz.mp4", "충전기", "Qi2", "전자제품"):
            self.assertIn(token, st)

    def test_missing_ext_meta_safe(self) -> None:
        # ext_meta 없음/None 도 빈 값으로 안전 처리.
        doc = asset_to_doc(self._row(ext_meta=None), channel="st")
        self.assertEqual(doc["summary"], "")
        self.assertEqual(doc["keywords"], [])
        self.assertEqual(doc["labels"], [])

    def test_non_list_keywords_ignored(self) -> None:
        # keywords/labels 가 리스트 아니면(스키마 위반) 빈 리스트로 방어.
        doc = asset_to_doc(
            self._row(ext_meta={"summary": "s", "keywords": "wrong"}), channel="st"
        )
        self.assertEqual(doc["keywords"], [])

    def test_zero_vector_omits_embedding(self) -> None:
        # 영벡터(퇴화 임베딩)는 embedding 필드를 아예 생략 → 텍스트만 색인(cosinesimil 거부 회피).
        doc = asset_to_doc(self._row(emb="[0.0,0.0,0.0]"), channel="st")
        self.assertNotIn("embedding", doc)
        self.assertEqual(doc["summary"], "무선 충전기 리뷰")  # 텍스트는 정상 색인

    def test_nonzero_vector_includes_embedding(self) -> None:
        # 한 성분이라도 0 이 아니면 embedding 포함(벡터 검색 대상).
        doc = asset_to_doc(self._row(emb="[0.0,0.1,0.0]"), channel="st")
        self.assertEqual(doc["embedding"], [0.0, 0.1, 0.0])


class TestIndexBody(unittest.TestCase):
    def test_knn_and_nori_mapping(self) -> None:
        body = build_index_body()
        props = body["mappings"]["properties"]
        # kNN 검색을 켠다.
        self.assertTrue(body["settings"]["index"]["knn"])
        # 임베딩은 knn_vector, 차원은 단일 출처 상수와 일치(헌법 6조·FR-005).
        self.assertEqual(props["embedding"]["type"], "knn_vector")
        self.assertEqual(props["embedding"]["dimension"], FIX_EMBEDDING_DIMENSION)
        self.assertEqual(props["embedding"]["method"]["space_type"], "cosinesimil")
        # 한국어 텍스트 필드는 nori 분석기.
        self.assertEqual(props["summary"]["analyzer"], "nori")
        self.assertEqual(props["search_text"]["analyzer"], "nori")
        # 메타 필터는 keyword.
        for k in ("asset_id", "modality", "domain_label", "status", "channel"):
            self.assertEqual(props[k]["type"], "keyword")

    def test_dim_override(self) -> None:
        # 차원은 인자로 덮어쓸 수 있다(테스트·향후 모델 교체 대비).
        body = build_index_body(dim=8)
        self.assertEqual(body["mappings"]["properties"]["embedding"]["dimension"], 8)


if __name__ == "__main__":
    unittest.main()
