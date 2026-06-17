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
from collections.abc import Callable
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.settings import get_current_settings
from src.database.postgres_util import PostgresUtil
from src.registry.lineage_persist import record_lineage
from src.relations.asset_candidates import (
    EmbeddingCandidate,
    EmbeddingKindFilter,
    find_embedding_candidates,
)
from src.relations.graph_persist import sync_graph_edges
from src.relations.llm_propose import parse_and_normalize_edges, propose_edges_json
from src.relations.path_signal import find_path_signal_candidates
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


def union_candidates(
    embedding_candidates: list[EmbeddingCandidate],
    path_candidates: list[EmbeddingCandidate],
) -> list[EmbeddingCandidate]:
    """임베딩 후보 ∪ 경로 신호 후보를 asset_id 기준 **dedup union**(C-3 임베딩 우선).

    규칙
        - 임베딩 후보를 **먼저** 둔다(이미 best_sim 내림차순·asset_id tiebreaker 로 결정적 정렬됨).
        - 경로 신호 후보 중 임베딩 후보와 **asset_id 가 겹치면 제외**한다 →
          임베딩 실측 emb_score 를 유지(C-3). path-only(emb_score=0.0 sentinel)는 그대로 합류.
        - 경로 신호 후보 내부 중복 asset_id 도 첫 항목만 유지(결정적).

    이 순서·dedup 규칙은 헌법 3조(재현성)를 위해 입력 정렬을 보존하는 안정 결합이다.
    union 총 후보 ≤ top_k + path_top_k(각 입력이 별도 LIMIT 됨, FR-013).
    """
    out: list[EmbeddingCandidate] = list(embedding_candidates)
    seen = {str(c["id"]) for c in embedding_candidates}
    for c in path_candidates:
        cid = str(c["id"])
        if cid in seen:  # 임베딩 후보와 겹침 → 임베딩 후보 유지(C-3), 경로 후보 버림
            continue
        seen.add(cid)
        out.append(c)
    return out


def target_emb_score_map(candidates: list[EmbeddingCandidate]) -> dict[str, float]:
    """후보의 ``{asset_id: emb_score}`` 맵 — 자동승인 AND 게이트(033 FR-003)에 전달.

    path-only 후보는 ``emb_score=0.0`` sentinel 을 그대로 유지(``emb_min>0`` 이면 자동승인 불가).
    union 에서 임베딩 후보가 우선이라 겹친 id 는 실측 emb_score 가 담긴다.
    """
    return {str(c["id"]): float(c["emb_score"]) for c in candidates}


def propose_relations_for_asset(
    db: PostgresUtil,
    source_asset_id: str,
    *,
    top_k: int | None = None,
    embedding_kind: EmbeddingKindFilter = "both",
    llm_fn: Callable[[str], dict[str, Any]] | None = None,  # 테스트용 주입(미주입=실 LLM 호출)
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
        emb_candidates = find_embedding_candidates(
            conn, source_asset_id=source_asset_id, top_k=k,
            embedding_kind=embedding_kind, min_sim=cfg.relation_min_sim,
        )
        # 레버 B(FR-004·005): 동일 폴더·파일명 stem 신호로 결정적 후보를 보강한다.
        # 임베딩 유사도가 min_sim 미만이라 emb_candidates 에 없던 same_series/derived_from/
        # references 후보가 여기서 합류한다. path_top_k 는 임베딩 top_k 와 별도 한도(C-2).
        path_candidates = find_path_signal_candidates(
            conn, source_asset_id=source_asset_id, limit=cfg.relation_path_top_k,
        )
        # C-3: 겹치면 임베딩 후보(실측 emb_score) 유지. 프롬프트·환각 화이트리스트 모두 이 union 을 쓴다.
        candidates = union_candidates(emb_candidates, path_candidates)
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
        # candidate_ids: 후보 집합 밖 target을 LLM 환각으로 간주해 sync_graph_edges에서 차단(FR-012).
        # candidates 는 임베딩 ∪ 경로 신호 union 이므로 allowed_target_ids 가 경로 신호 후보까지
        # **자동 확장**된다(FR-006) — 경로 신호로 추가된 target 에 LLM 이 단 엣지가 환각으로
        # 부당 차단되지 않는다. union 밖 target 거부 불변식은 그대로 유지된다.
        candidate_ids = frozenset(str(c["id"]) for c in candidates)

        # 신규 kind는 inactive로 먼저 등록 — 검토자가 active로 승인하기 전까지 그래프에 반영 안 됨.
        kinds_registered, kinds_skipped = register_new_relation_kinds(
            conn, edges=edges, active_kind_codes=active_codes)
        # 033 FR-003: 후보 emb_score 맵 + emb 하한을 persist 로 전달 → 자동승인 AND 게이트.
        # 기본 auto_approve_emb_min=0.0 이면 emb 변이 무력 → 현행과 동일(SC-001).
        edges_upserted, edges_skipped = sync_graph_edges(
            conn, source_asset_id=source_asset_id, edges=edges,
            allowed_target_ids=candidate_ids, auto_approve_min=cfg.relation_auto_approve_min,
            target_emb_scores=target_emb_score_map(candidates),
            auto_approve_emb_min=cfg.relation_auto_approve_emb_min)
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
