"""DB 연결·LLM 호출이 필요한 **관계 파이프라인 진입점** — asset_* 재배선판.

한 사이클
    1. ``asset`` + ``asset_metadata`` 에서 소스 요약(ext_meta.summary)·modality 로드.
    2. ``find_embedding_candidates`` 로 임베딩 후보 asset_id 집합 확보.
    3. DB에서 **활성 relation_kind** 카탈로그를 읽어 프롬프트 구성.
    4. LLM JSON 파싱 후, 신규 kind 만 inactive 등록(register_new_relation_kinds),
       ``sync_graph_edges`` 로 ``graph_edge`` 엣지를 upsert 한다.

트랜잭션
    ``PostgresUtil.execute_in_transaction`` 으로 후보 조회부터 kind 등록·엣지 기록까지 한 트랜잭션에서 처리한다.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.settings import get_current_settings
from src.database.postgres_util import PostgresUtil
from src.relations.asset_candidates import EmbeddingKindFilter, find_embedding_candidates
from src.registry.lineage_persist import record_lineage
from src.relations.graph_persist import sync_graph_edges
from src.relations.llm_propose import parse_and_normalize_edges, propose_edges_json
from src.relations.persist import register_new_relation_kinds
from src.relations.prompt import build_relation_proposal_prompt
from src.relations.relation_type_catalog import fetch_active_relation_kinds


def _fetch_source_row(conn: Connection[Any], asset_id: str) -> dict[str, Any] | None:
    """관계 판단에 쓸 소스 자산 한 행(id, 경로, modality, 요약). 없으면 None."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT a.asset_id,
                   a.fs_path,
                   a.modality,
                   COALESCE(m.ext_meta->>'summary', '') AS summary
            FROM asset a
            LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id
            WHERE a.asset_id = %s
            LIMIT 1
            """,
            (asset_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def propose_relations_for_asset(
    db: PostgresUtil,
    source_asset_id: str,
    *,
    top_k: int | None = None,
    embedding_kind: EmbeddingKindFilter = "both",
    llm_fn: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[int, int, int, int]:
    """
    후보 검색 → LLM JSON → 신규 kind inactive 등록 → ``graph_edge`` upsert.

    ``llm_fn`` 이 있으면 네트워크 대신 해당 함수로 프롬프트 문자열을 JSON으로 바꾼다(테스트용).

    Returns:
        (``kinds_registered``, ``kinds_skipped``, ``edges_upserted``, ``edges_skipped``)
    """
    cfg = get_current_settings()
    k = top_k if top_k is not None else cfg.relation_top_k

    def _run(conn: Connection[Any]) -> tuple[int, int, int, int]:
        src = _fetch_source_row(conn, source_asset_id)
        if src is None:
            # asset 테이블에 없는 ID — 조용히 (0,0,0,0) 반환(호출자 로그에서 확인).
            return 0, 0, 0, 0
        summary = str(src.get("summary") or "")
        candidates = find_embedding_candidates(
            conn, source_asset_id=source_asset_id, top_k=k,
            embedding_kind=embedding_kind, min_sim=cfg.relation_min_sim,
        )
        # 활성 relation_kind 목록을 프롬프트에 포함시켜 LLM이 통제 어휘 안에서 코드를 선택하게 한다.
        # 동시에 active_codes 집합을 만들어 신규 kind 등록 여부 판단에 재사용한다.
        kinds = fetch_active_relation_kinds(conn)
        prompt = build_relation_proposal_prompt(
            source_summary=summary,
            source_media_type=str(src.get("modality") or ""),
            candidates=candidates,
            relation_kinds_catalog=kinds,
        )
        raw = llm_fn(prompt) if llm_fn is not None else propose_edges_json(prompt)
        edges = parse_and_normalize_edges(raw)
        # active_codes: LLM이 반환한 kind 중 이미 카탈로그에 있는 것을 구분하는 기준.
        active_codes = frozenset(str(r["type_code"]) for r in kinds)
        # candidate_ids: 후보 집합 밖 target을 LLM 환각으로 간주해 sync_graph_edges에서 차단.
        candidate_ids = frozenset(str(c["id"]) for c in candidates)

        # 신규 kind는 inactive로 먼저 등록 — 검토자가 active로 승인하기 전까지 그래프에 반영 안 됨.
        kinds_registered, kinds_skipped = register_new_relation_kinds(
            conn, edges=edges, active_kind_codes=active_codes)
        edges_upserted, edges_skipped = sync_graph_edges(
            conn, source_asset_id=source_asset_id, edges=edges,
            allowed_target_ids=candidate_ids, auto_approve_min=cfg.relation_auto_approve_min)
        # 계보 기록: 이 자산에 대해 관계 제안이 실행되었음을 asset_lineage에 남긴다.
        # run_relations.py 가 재실행될 때 이중 기록이 생길 수 있으나, idempotent=False로
        # 트랜잭션 실패 시 롤백되어 반쪽 기록은 남지 않는다.
        record_lineage(
            conn,
            uuid.UUID(source_asset_id),
            activity="relations.proposed.v1",
            agent="llm_propose",
            generated={"edges_upserted": edges_upserted, "edges_skipped": edges_skipped,
                       "kinds_registered": kinds_registered},
            payload={"top_k": k, "embedding_kind": embedding_kind},
        )
        return kinds_registered, kinds_skipped, edges_upserted, edges_skipped

    return db.execute_in_transaction(_run, idempotent=False)
