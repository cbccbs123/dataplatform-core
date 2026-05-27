"""멀티모달 미디어 간 **관계 카탈로그(LLM 보조)** 패키지.

데이터 흐름(요약)
    1. ``find_embedding_candidates``: 소스 자산과 채널이 맞는 ``asset_embedding`` 끼리 pgvector 유사도 상위 k 후보.
    2. ``build_relation_proposal_prompt`` + ``propose_edges_json``: DB 카탈로그·후보 메타를 LLM에 넘겨 JSON 엣지 수신.
    3. ``parse_and_normalize_edges`` / ``schema``: LLM 출력 정규화(관계 코드·토피 필드).
    4. ``sync_relation_catalog_from_llm_edges``: 허용 코드·sanitize 규칙으로 카탈로그(``relation_kind`` /
       ``relation_topic_parent`` / ``relation_subtopic`` / ``relation_type``)를 갱신한 뒤,
       ``sync_graph_edges`` 로 ``graph_edge`` 엣지를 upsert 한다. 진입점은 ``propose_relations_for_asset``.

의존성
    ``psycopg``, OpenAI 클라이언트 등 **무거운 의존성**은 ``asset_entry``·``persist``·``llm_propose`` 쪽에만 두고,
    ``from src.relations import X`` 는 ``__getattr__`` 로 **지연 import** 한다(순환 참조·기동 비용 완화).

테스트
    DB 없이 돌리려면 ``schema``, ``resolve_relation_type`` 등을 직접 import 한다.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "propose_relations_for_asset",
    "find_embedding_candidates",
    "build_relation_proposal_prompt",
    "parse_and_normalize_edges",
    "propose_edges_json",
    "sync_relation_catalog_from_llm_edges",
    "resolve_relation_type_code",
]


def __getattr__(name: str) -> Any:
    """패키지 속성 지연 로드: ``import src.relations as R; R.propose_relations_for_asset`` 형태만 지원."""
    if name == "propose_relations_for_asset":
        from src.relations.asset_entry import propose_relations_for_asset as fn

        return fn
    if name == "find_embedding_candidates":
        from src.relations.asset_candidates import find_embedding_candidates as fn

        return fn
    if name == "build_relation_proposal_prompt":
        from src.relations.prompt import build_relation_proposal_prompt as fn

        return fn
    if name == "parse_and_normalize_edges":
        from src.relations.llm_propose import parse_and_normalize_edges as fn

        return fn
    if name == "propose_edges_json":
        from src.relations.llm_propose import propose_edges_json as fn

        return fn
    if name == "sync_relation_catalog_from_llm_edges":
        from src.relations.persist import sync_relation_catalog_from_llm_edges as fn

        return fn
    if name == "resolve_relation_type_code":
        from src.relations.resolve_relation_type import resolve_relation_type_code as fn

        return fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
