"""관계 HITL 검토 — proposed 엣지 승인/반려 + relation_kind 승격(inactive→active)."""
from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row


def list_proposed_edges(conn: Connection[Any], *, limit: int = 100) -> list[dict[str, Any]]:
    """검토 대기(proposed) 엣지를 신뢰도 높은 순으로. status='active' 필터는 소비자 몫."""
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
    """proposed 엣지만 status 로 확정(이미 결정된 엣지는 변경 안 함). 1행 갱신 시 True."""
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
    """LLM 제안으로 쌓인 inactive relation_kind 를 active 로 승격(어휘 거버넌스). reviewer 는 lineage 기록용."""
    _ = reviewer
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE relation_kind SET status='active' WHERE kind_code=%s AND status='inactive'",
            (kind_code,),
        )
        return cur.rowcount == 1
