"""공용 그래프 영속화 — node(asset) 보장 + graph_edge upsert (asset_relation 흡수판).

엣지 종류는 기존 relation_type 카탈로그(relation_kind×subtopic)를 재사용한다.
sync_asset_relation_edges 와 동일한 검증(후보 집합·자기참조·코드 정규화·relation_type 해소)을 따르되,
양 끝 asset-노드를 보장하고 graph_edge 에 upsert 한다.
"""
from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection

from src.database.ids import uuid7_str
from src.relations.relation_type_catalog import fetch_relation_type_id_for_normalized_edge
from src.relations.schema import coerce_topic_fields_mvp


def _as_uuid_str(value: Any) -> str | None:
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def ensure_asset_node(conn: Connection[Any], asset_id: str) -> str:
    """asset_id 의 asset-노드를 보장하고 node_id(UUID str) 반환(SELECT→없으면 INSERT)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT node_id FROM node WHERE node_kind = 'asset' AND asset_id = %s LIMIT 1",
            (asset_id,),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row[0])
        nid = uuid7_str()
        # 동시성 안전: 부분 유니크 인덱스(uq_node_asset)와 ON CONFLICT 로 경쟁 INSERT 흡수.
        cur.execute(
            """
            INSERT INTO node (node_id, node_kind, asset_id) VALUES (%s, 'asset', %s)
            ON CONFLICT (asset_id) WHERE node_kind = 'asset' DO NOTHING
            RETURNING node_id
            """,
            (nid, asset_id),
        )
        inserted = cur.fetchone()
        if inserted is not None:
            return str(inserted[0])
        # 다른 트랜잭션이 먼저 INSERT(충돌) → 기존 노드 재조회
        cur.execute(
            "SELECT node_id FROM node WHERE node_kind = 'asset' AND asset_id = %s LIMIT 1",
            (asset_id,),
        )
        return str(cur.fetchone()[0])


def sync_graph_edges(
    conn: Connection[Any],
    *,
    source_asset_id: str,
    edges: list[dict[str, Any]],
    allowed_target_ids: frozenset[str],
) -> tuple[int, int]:
    """후보 집합 안의 타깃만 graph_edge 에 기록(양 끝 asset-노드 보장). Returns (upserted, skipped)."""
    allowed = frozenset(str(t) for t in allowed_target_ids)
    upserted = 0
    skipped = 0
    src_node = ensure_asset_node(conn, source_asset_id)
    for edge in edges:
        tid = _as_uuid_str(edge.get("target_media_item_id"))
        if tid is None or tid not in allowed or tid == source_asset_id:
            skipped += 1
            continue
        code = edge.get("relation_type_code")
        if not code or not str(code).strip():
            skipped += 1
            continue
        kind_code = str(code).strip().lower()
        topic_ko, subtopic_ko, topic_en, subtopic_en, _ = coerce_topic_fields_mvp(edge)
        rtid = fetch_relation_type_id_for_normalized_edge(
            conn,
            kind_code=kind_code,
            topic_ko=topic_ko,
            subtopic_ko=subtopic_ko,
            topic_en=topic_en,
            subtopic_en=subtopic_en,
        )
        if rtid is None:
            skipped += 1
            continue
        conf = edge.get("confidence")
        try:
            conf_f = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            conf_f = None
        reason = str(edge.get("reason") or "").strip() or None
        dst_node = ensure_asset_node(conn, tid)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO graph_edge (edge_id, src_node, dst_node, relation_type_id, confidence, reason, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'active')
                ON CONFLICT (src_node, dst_node, relation_type_id)
                DO UPDATE SET confidence = EXCLUDED.confidence, reason = EXCLUDED.reason,
                              status = 'active', updated_at = now()
                """,
                (uuid7_str(), src_node, dst_node, rtid, conf_f, reason),
            )
        upserted += 1
    return upserted, skipped
