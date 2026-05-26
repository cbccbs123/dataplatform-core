"""``asset_relation`` 엣지 테이블 upsert (카탈로그 ``relation_type_id`` 참조) — asset_* 재배선판.

OLD ``media_relation_persist.py`` 와의 차이
    media_relation→asset_relation, 타깃 id 가 UUID(str), PK ``relation_id`` 를 앱에서 uuid7 로 생성.
    카탈로그 조회·토픽 정규화 로직은 동일하게 재사용한다.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection

from src.database.ids import uuid7_str
from src.relations.relation_type_catalog import fetch_relation_type_id_for_normalized_edge
from src.relations.schema import coerce_topic_fields_mvp


def _as_uuid_str(value: Any) -> str | None:
    """UUID 로 파싱되면 정규 문자열, 아니면 None."""
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        return None


def sync_asset_relation_edges(
    conn: Connection[Any],
    *,
    source_asset_id: str,
    edges: list[dict[str, Any]],
    allowed_target_ids: frozenset[str],
) -> tuple[int, int]:
    """
    임베딩 후보 집합 안의 타깃만 ``asset_relation`` 에 기록한다.

    ``relation_type_id`` 는 ``sync_relation_catalog_from_llm_edges`` 이후 DB에 존재하는
    (kind + 토픽 번들) 행을 조회한다(inactive 포함). PK ``relation_id`` 는 신규 INSERT 시
    ``uuid7()`` 로 생성하되, ON CONFLICT(edge 유니크) 시에는 기존 행을 UPDATE 한다.

    Returns:
        (``upserted``, ``skipped``)
    """
    allowed = frozenset(str(t) for t in allowed_target_ids)
    upserted = 0
    skipped = 0
    for edge in edges:
        tid = _as_uuid_str(edge.get("target_media_item_id"))
        if tid is None:
            skipped += 1
            continue
        if tid not in allowed:
            skipped += 1
            continue
        if tid == source_asset_id:
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
                INSERT INTO asset_relation (
                    relation_id,
                    source_asset_id,
                    target_asset_id,
                    relation_type_id,
                    confidence,
                    reason,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, 'active')
                ON CONFLICT (source_asset_id, target_asset_id, relation_type_id)
                DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    reason = EXCLUDED.reason,
                    status = 'active',
                    updated_at = now()
                """,
                (
                    uuid7_str(),
                    source_asset_id,
                    tid,
                    rtid,
                    conf_f,
                    reason,
                ),
            )
        upserted += 1
    return upserted, skipped
