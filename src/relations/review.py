"""관계 HITL 검토 — proposed 엣지 승인/반려 + relation_kind 승격(inactive→active).

검토 큐 설계
    별도 큐 테이블 없이 ``graph_edge.status = 'proposed'`` 자체가 검토 큐다.
    ``list_proposed_edges`` 로 조회하고, ``approve_edge``/``reject_edge`` 로 상태를 전환한다.

멱등·반려 부활 방지 계약
    ``_decide_edge`` 의 ``WHERE edge_id = %s AND status = 'proposed'`` 가드가 핵심이다.
    이미 active 또는 rejected 인 엣지는 갱신 대상에서 제외되므로 rowcount == 0 이 되어
    False 를 반환한다. 소비자는 이 반환값으로 "이미 결정된 엣지" 여부를 판단한다.

관계 어휘 거버넌스
    ``promote_relation_kind`` 는 LLM 이 제안해 inactive 로 쌓인 kind 를 active 로 올리는
    HITL 경로다. 반대 방향(active→inactive 강등)은 현재 미구현 — 직접 SQL 로 처리한다.
"""
from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row


def list_proposed_edges(conn: Connection[Any], *, limit: int = 100) -> list[dict[str, Any]]:
    """검토 대기(proposed) 엣지를 신뢰도 높은 순으로. status='active' 필터는 소비자 몫.

    ``topic`` 컬럼은 jsonb(topic_ko/en, subtopic_ko/en)다. 소비자가 필요한 키를 꺼내 쓴다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT e.edge_id, e.src_node, e.dst_node, rk.kind_code,
                   e.confidence, e.reason, e.topic
            FROM graph_edge e
            JOIN relation_kind rk ON rk.relation_kind_id = e.relation_kind_id
            WHERE e.status = 'proposed'
            ORDER BY e.confidence DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def _decide_edge(conn: Connection[Any], *, edge_id: str, reviewer: str, status: str) -> bool:
    """proposed 엣지만 status 로 확정(이미 결정된 엣지는 변경 안 함). 1행 갱신 시 True.

    ``AND status = 'proposed'`` 가드 목적
        - 이미 active 인 엣지를 실수로 재반려하는 사고를 차단한다.
        - 이미 rejected 인 엣지가 재승인 요청으로 부활하는 것을 막는다.
        - rowcount == 1 확인으로 "내가 이 결정을 실제로 수행했는가"를 반환한다.
          rowcount == 0 은 "엣지 없음" 또는 "이미 결정됨" 두 경우를 포함한다.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE graph_edge
            SET status = %s, reviewed_by = %s, reviewed_at = now(), updated_at = now()
            WHERE edge_id = %s AND status = 'proposed'
            """,
            (status, reviewer, edge_id),
        )
        return cur.rowcount == 1


def approve_edge(conn: Connection[Any], *, edge_id: str, reviewer: str) -> bool:
    return _decide_edge(conn, edge_id=edge_id, reviewer=reviewer, status="active")


def reject_edge(conn: Connection[Any], *, edge_id: str, reviewer: str) -> bool:
    return _decide_edge(conn, edge_id=edge_id, reviewer=reviewer, status="rejected")


def promote_relation_kind(conn: Connection[Any], *, kind_code: str, reviewer: str) -> bool:
    """LLM 제안으로 쌓인 inactive relation_kind 를 active 로 승격(어휘 거버넌스). reviewer 는 lineage 기록용.

    ``AND status = 'inactive'`` 가드
        이미 active 인 kind 를 중복 승격해도 rowcount == 0 이 되어 False 를 반환한다(멱등).

    reviewer 미저장 이유
        현재 relation_kind 스키마에 reviewed_by 컬럼이 없다.
        ``_ = reviewer`` 는 시그니처는 유지하면서 미래 lineage 컬럼 추가를 위한 자리 표시자다.
    """
    _ = reviewer  # 향후 relation_kind.reviewed_by 컬럼 추가 시 여기에 저장
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE relation_kind SET status='active' WHERE kind_code=%s AND status='inactive'",
            (kind_code,),
        )
        return cur.rowcount == 1
