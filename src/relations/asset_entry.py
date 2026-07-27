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

import logging
import uuid
from collections.abc import Callable
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.settings import get_current_settings
from src.database.lineage_persist import record_lineage
from src.database.postgres_util import PostgresUtil
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
from src.topic.asset_topic_query import fetch_asset_topic

_LOG = logging.getLogger(__name__)


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

    Args:
        embedding_candidates: ``find_embedding_candidates`` 결과(유사도 정렬됨). 그대로 앞에 온다.
        path_candidates: ``find_path_signal_candidates`` 결과(경로·파일명 신호). ``emb_score`` 는
            실측값이 아닌 ``0.0`` sentinel 이라 자동승인 emb 게이트를 통과하지 못한다.

    Returns:
        중복 없는 후보 리스트(입력 순서 보존). 둘 다 비면 빈 리스트.
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

    Args:
        candidates: ``union_candidates`` 결과(임베딩 ∪ 경로 신호).

    Returns:
        ``{asset_id: emb_score}``. 같은 id 가 겹치면 뒤 항목이 덮어쓰지만, union 이 이미 dedup 하므로
        실제로는 발생하지 않는다.
    """
    return {str(c["id"]): float(c["emb_score"]) for c in candidates}


def propose_relations_for_asset(
    db: PostgresUtil,
    source_asset_id: str,
    *,
    top_k: int | None = None,
    embedding_kind: EmbeddingKindFilter = "st",  # 036: bge-only 기본(단일 st_bge 공간·척도 일관)
    llm_fn: Callable[[str], dict[str, Any]] | None = None,  # 테스트용 주입(미주입=실 LLM 호출)
) -> tuple[int, int, int, int]:
    """한 자산의 관계를 제안해 그래프에 반영한다 — 후보 검색 → LLM → kind 등록 → 엣지 upsert.

    **DB에 쓴다**(단일 트랜잭션): ``relation_kind`` 신규 코드 inactive 등록 · ``graph_edge`` upsert ·
    ``asset_lineage`` 에 ``relations.proposed.v1`` 기록. 트랜잭션은 ``idempotent=False`` 로 실행되어
    **재시도하지 않는다** — 부분 적용된 쓰기를 다시 돌리면 중복 반영될 수 있어서다. 실패 시 전부 롤백.

    LLM 호출이 한 번 일어난다(``llm_fn`` 미주입 시). 소스 자산이 없거나 **자기주제가 미부여**면
    LLM·후보 검색을 아예 건너뛰고 ``(0, 0, 0, 0)`` 을 돌려준다.

    Args:
        source_asset_id: 관계를 제안할 기준 자산.
        top_k: 임베딩 후보 상한. ``None`` 이면 설정값(``relations.top_k``)을 쓴다.
            경로 신호 후보는 이 값과 **별도 한도**(``relations.path_top_k``)를 가진다.
        embedding_kind: 후보 검색에 쓸 채널(``st``/``clip``/``both``). 기본은 텍스트 단일 공간.
        llm_fn: **테스트용 주입 seam** — 미주입이면 운영 LLM(``propose_edges_json``)을 호출한다.
            주입 시 프롬프트 문자열을 받아 제안 JSON(dict)을 돌려주는 함수여야 한다.

    Returns:
        ``(kinds_registered, kinds_skipped, edges_upserted, edges_skipped)``.
        자산 없음·주제 미부여면 ``(0, 0, 0, 0)``.
    """
    cfg = get_current_settings()
    k = top_k if top_k is not None else cfg.relations.top_k

    def _run(conn: Connection[Any]) -> tuple[int, int, int, int]:
        """한 트랜잭션 안에서 실행되는 본체 — 조회·LLM·쓰기가 모두 여기서 일어난다.

        ``execute_in_transaction`` 에 넘겨져 커넥션을 받는다. 중간에 예외가 나면 이 함수가 남긴
        쓰기(kind 등록·엣지·계보)는 통째로 롤백된다.
        """
        src = _fetch_source_row(conn, source_asset_id)
        if src is None:
            # asset 테이블에 없는 ID — 조용히 (0,0,0,0) 반환(호출자 로그에서 확인).
            return 0, 0, 0, 0
        # 066 FR-102/103: 소스가 미부여(asset_topic 행 없음·무내용)면 관계 생성을 스킵한다.
        #   후보검색·LLM 을 아예 호출하지 않고 (0,0,0,0) 을 반환 → 상위 run_relations 의
        #   _record_resolution 이 기존 decide_resolution_status(0엣지·무예외→isolated)로 종결한다.
        #   (스캔·상태기계는 불변 — 여기선 0엣지 반환만 보장해 재시도·LLM 낭비를 없앤다.)
        #   FR-201 재사용: 부여됐으면 이 조회 결과에서 소스 주제(topic_ko/subtopic_ko)를 얻어 프롬프트에 싣는다.
        source_topics = fetch_asset_topic(conn, source_asset_id)
        if not source_topics:
            _LOG.info("미부여 소스 관계 스킵: %s", source_asset_id)
            return 0, 0, 0, 0
        source_topic = {
            "topic_ko": source_topics[0].get("topic_ko"),
            "subtopic_ko": source_topics[0].get("subtopic_ko"),
        }
        summary = str(src.get("summary") or "")
        emb_candidates = find_embedding_candidates(
            conn, source_asset_id=source_asset_id, top_k=k,
            embedding_kind=embedding_kind, min_sim=cfg.relations.min_sim,
        )
        # 레버 B(FR-004·005): 동일 폴더·파일명 stem 신호로 결정적 후보를 보강한다.
        # 임베딩 유사도가 min_sim 미만이라 emb_candidates 에 없던 same_series/derived_from/
        # references 후보가 여기서 합류한다. path_top_k 는 임베딩 top_k 와 별도 한도(C-2).
        path_candidates = find_path_signal_candidates(
            conn, source_asset_id=source_asset_id, limit=cfg.relations.path_top_k,
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
            source_topic=source_topic,  # 066 FR-202: 소스 자기주제를 soft 신호로 전달.
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
        # collect: upsert 된 관계 쌍(상대자산·관계유형·신뢰도·status)을 계보에 남기기 위해 모은다(013).
        upserted_pairs: list[dict[str, Any]] = []
        edges_upserted, edges_skipped = sync_graph_edges(
            conn, source_asset_id=source_asset_id, edges=edges,
            allowed_target_ids=candidate_ids, auto_approve_min=cfg.relations.auto_approve_min,
            target_emb_scores=target_emb_score_map(candidates),
            auto_approve_emb_min=cfg.relations.auto_approve_emb_min,
            collect=upserted_pairs)
        # 계보 기록: 이 자산에 대해 관계 제안이 실행되었음을 asset_lineage에 남긴다.
        # 078: lineage_persist 를 코어(src.database)로 이동 — 코어 relations 가 ingest(파이프라인)를
        # 역참조하던 것을 해소한다(record_lineage 는 코어 표 asset_lineage 쓰기·ingest/relations 공용이라 코어 소속).
        # 호출·원자성은 불변(idempotent=False로 트랜잭션 실패 시 롤백되어 반쪽 기록 없음).
        # generated.edges: 생성된 관계 쌍을 target_asset_id ASC(동점 kind_code ASC)로 결정적 정렬(헌법 3조).
        record_lineage(
            conn,
            uuid.UUID(source_asset_id),
            activity="relations.proposed.v1",
            agent="llm_propose",
            generated={"edges_upserted": edges_upserted, "edges_skipped": edges_skipped,
                       "kinds_registered": kinds_registered,
                       "edges": sorted(upserted_pairs,
                                       key=lambda e: (e["target_asset_id"], e["kind_code"]))},
            payload={"top_k": k, "embedding_kind": embedding_kind},
        )
        return kinds_registered, kinds_skipped, edges_upserted, edges_skipped

    return db.execute_in_transaction(_run, idempotent=False)
