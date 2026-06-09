"""PostgreSQL(`asset_*`) → OpenSearch 데이터 동기화 (검색 엔진 도입 — spec 020).

PG 는 **읽기 전용**(SELECT 만), OpenSearch 에만 색인을 쓴다(CQRS — 원본 DB 무수정, 헌법 6조).

설계
    - **자산 1건 = OpenSearch 문서 1개**. 임베딩은 활성 채널 청크들의 **평균 풀링**(`avg(embedding)`)
      한 벡터를 `knn_vector` 로 색인한다 — 019 측정에서 평균 집계가 MAX 보다 검색 품질이 좋았고,
      자산당 단일 벡터라 색인·질의가 단순하다(청크별 색인은 후속 선택지).
    - **하이브리드 한 인덱스**: 텍스트(summary·keywords·labels·file_name)는 한국어 형태소 분석기
      `nori` 로 BM25, 임베딩은 `knn_vector`(코사인). 메타(modality·domain_label·status·channel)는
      keyword 필터.

이 모듈의 **순수 함수**(`build_index_body`·`asset_to_doc`·`parse_vector`)는 DB·OS·opensearch-py
없이 결정적으로 동작하며 단위 게이트에서 항상 검증된다. **IO 함수**(get_client·ensure_index·
색인 실행)는 후속 그룹(G2)에서 추가하며, opensearch-py·psycopg 의존은 모듈 상단이 아니라
**해당 함수 내부에서 지연 import** 한다 — 플래그 off(미도입) 환경의 순수성을 보존하기 위함이다.
"""

from __future__ import annotations

from typing import Any

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION


def parse_vector(value: Any) -> list[float]:
    """pgvector 반환값(리스트 또는 ``'[0.1,0.2,...]'`` 문자열)을 float 리스트로 정규화(순수)."""
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    s = str(value).strip()
    inner = s[1:-1] if s.startswith("[") and s.endswith("]") else s
    return [float(x) for x in inner.split(",") if x.strip()]


def _basename(uri: str) -> str:
    """경로/URI 에서 파일명만 추출한다(결정적·순수). 쿼리·프래그먼트는 제거."""
    if not uri:
        return ""
    tail = uri.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail.split("?", 1)[0].split("#", 1)[0] or tail


def build_index_body(*, dim: int = FIX_EMBEDDING_DIMENSION) -> dict[str, Any]:
    """자산 인덱스 settings+mappings(순수). nori 한국어 분석기 + knn_vector(코사인).

    ``index.knn=true`` 로 kNN 검색을 켜고, 텍스트 필드는 ``analyzer='nori'``(analysis-nori 플러그인
    내장 분석기)로 한국어 형태소 BM25 를 쓴다. 임베딩은 lucene HNSW + cosinesimil. 차원은 단일
    출처 ``FIX_EMBEDDING_DIMENSION``(1536D, 헌법 6조)을 기본값으로 따른다.
    """
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                "asset_id": {"type": "keyword"},
                "modality": {"type": "keyword"},
                "domain_label": {"type": "keyword"},
                "status": {"type": "keyword"},
                "channel": {"type": "keyword"},
                "file_name": {
                    "type": "text",
                    "analyzer": "nori",
                    "fields": {"raw": {"type": "keyword"}},
                },
                "fs_uri": {"type": "keyword"},
                "summary": {"type": "text", "analyzer": "nori"},
                "keywords": {"type": "text", "analyzer": "nori"},
                "labels": {"type": "keyword"},
                "search_text": {"type": "text", "analyzer": "nori"},
                "chunk_count": {"type": "integer"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }


def asset_to_doc(row: dict[str, Any], channel: str) -> dict[str, Any]:
    """PG 행(asset+metadata+평균임베딩) → OpenSearch 문서(순수·결정적).

    ``search_text`` 는 summary·file_name·keywords·labels 를 합쳐 BM25 대상으로 둔다.
    ext_meta 가 None/비-리스트(스키마 위반)여도 빈 값으로 안전 처리한다.
    """
    ext = row.get("ext_meta") or {}
    file_name = _basename(str(row.get("fs_path") or ""))
    summary = str(ext.get("summary") or "")
    keywords = ext.get("keywords") if isinstance(ext.get("keywords"), list) else []
    labels = ext.get("labels") if isinstance(ext.get("labels"), list) else []

    parts = [summary, file_name]
    parts += [str(k) for k in keywords]
    parts += [str(label) for label in labels]
    search_text = " ".join(p for p in parts if p)

    doc = {
        "asset_id": str(row["asset_id"]),
        "modality": row.get("modality"),
        "domain_label": row.get("domain_label"),
        "status": row.get("status"),
        "channel": channel,
        "file_name": file_name,
        "fs_uri": str(row.get("fs_path") or ""),
        "summary": summary,
        "keywords": [str(k) for k in keywords],
        "labels": [str(label) for label in labels],
        "search_text": search_text,
        "chunk_count": int(row.get("chunk_count") or 0),
    }
    # 영벡터(퇴화 임베딩 — 빈 STT 등)는 cosinesimil knn 이 거부하므로 embedding 필드를 **생략**한다.
    # 해당 자산은 텍스트(BM25)로만 검색되고 벡터 검색 대상에서만 빠진다(색인 실패 대신 우아한 처리).
    vec = parse_vector(row["emb"])
    if any(x != 0.0 for x in vec):
        doc["embedding"] = vec
    return doc
