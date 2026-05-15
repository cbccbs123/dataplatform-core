"""DB 연결·LLM 호출이 필요한 **관계 파이프라인 진입점**.

한 사이클
    1. ``media_items`` 에서 소스 요약·타입 로드.
    2. ``find_embedding_candidates`` 로 임베딩 후보 id 집합 확보.
    3. DB에서 **활성 relation_type** 카탈로그·dedup용 전체 목록을 읽어 프롬프트 구성.
    4. LLM JSON 파싱 후, DB **활성** 프롬프트 카탈로그에 나온 ``type_code`` 집합으로 **허용 관계 코드**를 만들고
       ``sync_relation_catalog_from_llm_edges`` 로 **카탈로그**를 갱신한 뒤,
       ``sync_media_relation_edges`` 로 **``media_relation`` 엣지**를 upsert 한다.

트랜잭션
    ``PostgresUtil.execute_in_transaction`` 으로 후보 조회부터 카탈로그·엣지 기록까지 한 트랜잭션에서 처리한다.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.settings import get_current_settings
from src.database.postgres_util import PostgresUtil
from src.relations.candidates import EmbeddingKindFilter, find_embedding_candidates
from src.relations.llm_propose import parse_and_normalize_edges, propose_edges_json
from src.relations.media_relation_persist import sync_media_relation_edges
from src.relations.persist import sync_relation_catalog_from_llm_edges
from src.relations.prompt import build_relation_proposal_prompt
from src.relations.relation_type_catalog import (
    fetch_llm_prompt_relation_types,
    fetch_relation_types_for_prompt_dedup,
)


def _fetch_source_row(conn: Connection[Any], media_item_id: int) -> dict[str, Any] | None:
    """관계 판단에 쓸 소스 ``media_items`` 한 행(id, uri, type, metadata). 없으면 None."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, file_uri, media_type, metadata
            FROM media_items
            WHERE id = %s
            LIMIT 1
            """,
            (media_item_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def propose_relations_for_item(
    db: PostgresUtil,
    source_media_item_id: int,
    *,
    top_k: int | None = None,
    embedding_kind: EmbeddingKindFilter = "both",
    llm_fn: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[int, int, int, int]:
    """
    후보 검색 → LLM JSON → 카탈로그 동기화 → ``media_relation`` upsert.

    ``llm_fn`` 이 있으면 네트워크 대신 해당 함수로 프롬프트 문자열을 JSON으로 바꾼다(테스트용).

    ``allowed`` (``sync_relation_catalog_from_llm_edges`` 로 전달되는 ``llm_prompt_type_codes``)
        DB에서 읽은 **활성** ``relation_type`` 프롬프트 카탈로그에 등장하는 ``type_code``(= ``relation_kind.kind_code``) 전부다.
        프롬프트에 없는 코드는 비활성 번들 등록 분기(``sanitize`` 통과 시)로만 처리된다.

    Returns:
        (``catalog_synced``, ``catalog_skipped``, ``edges_upserted``, ``edges_skipped``)
    """
    cfg = get_current_settings()
    k = top_k if top_k is not None else cfg.relation_top_k

    def _run(conn: Connection[Any]) -> tuple[int, int]:
        src = _fetch_source_row(conn, source_media_item_id)
        if src is None:
            return 0, 0, 0, 0
        meta = src.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        summary = str((meta or {}).get("summary") or "")
        # 임베딩 후보: LLM에 넣을 target 목록이자, 이후 "후보 집합 안의 id만 확정" 검증에 사용된다.
        candidates = find_embedding_candidates(
            conn,
            source_media_item_id=source_media_item_id,
            top_k=k,
            embedding_kind=embedding_kind,
        )
        # 활성 relation_type 만 카탈로그 JSON에 넣음. dedup 블록은 비활성 포함(중복 코드 재제안 방지).
        catalog = fetch_llm_prompt_relation_types(conn)
        type_registry = fetch_relation_types_for_prompt_dedup(conn)
        prompt = build_relation_proposal_prompt(
            source_summary=summary,
            source_media_type=str(src.get("media_type") or ""),
            candidates=candidates,
            relation_types_catalog=catalog,
            existing_type_registry=type_registry,
        )
        if llm_fn is not None:
            raw = llm_fn(prompt)
        else:
            raw = propose_edges_json(prompt)
        edges = parse_and_normalize_edges(raw)
        # 프롬프트에 실린 활성 카탈로그 type_code 집합 → persist 가 기존 kind 경로로 처리할 코드.
        allowed = frozenset(str(r["type_code"]) for r in catalog)
        candidate_ids = frozenset(int(c["id"]) for c in candidates)

        catalog_synced, catalog_skipped = sync_relation_catalog_from_llm_edges(
            conn,
            source_media_item_id=source_media_item_id,
            edges=edges,
            llm_prompt_type_codes=allowed,
        )
        edges_upserted, edges_skipped = sync_media_relation_edges(
            conn,
            source_media_item_id=source_media_item_id,
            edges=edges,
            allowed_target_ids=candidate_ids,
        )
        return catalog_synced, catalog_skipped, edges_upserted, edges_skipped

    return db.execute_in_transaction(_run, idempotent=False)
