"""검색 패키지 공개부 — OpenSearch 전용 정리(037) 후 질의 임베더만 re-export.

037 그룹 B 에서 PG FTS/벡터 검색(``media_search``)을 제거하면서 PG 검색 함수
(``search_media_*``) re-export 도 함께 걷어냈다. 검색 read path 는 ``opensearch_search`` 단일
경로이며, 호출부는 ``search_service.search_hybrid`` 를 통해 접근한다. 채널/차원 상수
(``EMBEDDING_KIND_*``·``DEFAULT_CLIP_MODEL_NAME``·``FIX_EMBEDDING_DIMENSION``)는 소비처가 전부
``src.config.embedding_constants`` 를 직접 import 하므로(이 패키지 재수출 소비 0) 069 US-F 에서
재수출을 제거했다. 여기서는 질의 임베더(``embed_query_for_media_search``)만 노출한다.
"""

from src.search.query_embed import embed_query_for_media_search

__all__ = [
    "embed_query_for_media_search",
]
