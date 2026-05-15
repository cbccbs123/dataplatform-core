"""``media_relation`` 엣지 테이블 upsert (카탈로그 ``relation_type_id`` 참조)."""

from __future__ import annotations

from typing import Any

from psycopg import Connection

from src.relations.relation_type_catalog import fetch_relation_type_id_for_normalized_edge
from src.relations.schema import coerce_topic_fields_mvp


def sync_media_relation_edges(
    conn: Connection[Any],
    *,
    source_media_item_id: int,
    edges: list[dict[str, Any]],
    allowed_target_ids: frozenset[int],
) -> tuple[int, int]:
    """
    임베딩 후보 집합 안의 타깃만 ``media_relation`` 에 기록한다.

    ``relation_type_id`` 는 ``sync_relation_catalog_from_llm_edges`` 이후 DB에 존재하는
    (kind + 토피 번들) 행을 조회한다(inactive 포함).

    Returns:
        (``upserted``, ``skipped``)
    """
    upserted = 0
    skipped = 0
    for edge in edges:
        tid = edge.get("target_media_item_id")
        if tid is None:
            skipped += 1
            continue
        try:
            tid_int = int(tid)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if tid_int not in allowed_target_ids:
            skipped += 1
            continue
        if tid_int == source_media_item_id:
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
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO media_relation (
                    source_media_item_id,
                    target_media_item_id,
                    relation_type_id,
                    confidence,
                    reason,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, 'active')
                ON CONFLICT (source_media_item_id, target_media_item_id, relation_type_id)
                DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    reason = EXCLUDED.reason,
                    status = 'active',
                    updated_at = now()
                """,
                (
                    source_media_item_id,
                    tid_int,
                    rtid,
                    conf_f,
                    reason,
                ),
            )
        upserted += 1
    return upserted, skipped
