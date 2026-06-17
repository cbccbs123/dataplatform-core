"""검색 패키지 공개부 — OpenSearch 전용 정리(037) 후 공유 상수·질의 임베더만 re-export.

037 그룹 B 에서 PG FTS/벡터 검색(``media_search``)을 제거하면서 PG 검색 함수
(``search_media_*``) re-export 도 함께 걷어냈다. 검색 read path 는 ``opensearch_search`` 단일
경로이며, 호출부는 ``search_service.search_hybrid`` 를 통해 접근한다. 여기서는 적재·검색·관계가
공유하는 채널 상수(``EMBEDDING_KIND_*``)와 질의 임베더(``embed_query_for_media_search``)만 노출한다.
"""

from src.config.embedding_constants import (
    DEFAULT_CLIP_MODEL_NAME,
    EMBEDDING_KIND_CLIP,
    EMBEDDING_KIND_ST,
    FIX_EMBEDDING_DIMENSION,
)
from src.search.query_embed import embed_query_for_media_search

__all__ = [
    "DEFAULT_CLIP_MODEL_NAME",
    "EMBEDDING_KIND_CLIP",
    "EMBEDDING_KIND_ST",
    "FIX_EMBEDDING_DIMENSION",
    "embed_query_for_media_search",
]
