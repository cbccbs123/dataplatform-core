"""LLM 엣지에서 **관계 카탈로그**와 동일 트랜잭션에서 쓰일 ``relation_type`` 행을 동기화한다.

MVP 확장: ``media_relation`` 엣지는 ``entry.propose_relations_for_item`` 이 ``sync_media_relation_edges`` 로 따로 기록한다.

처리(엣지별)
    1. 토피 튜플 정규화 후 ``relation_subtopic_id`` 확보(대주제·서브토픽 2단 테이블).
    2. ``relation_type_code`` 정규화 실패(None) → 스킵.
    3. 코드가 ``llm_prompt_type_codes`` 밖이면 ``sanitize`` 통과 시에만
       비활성 ``relation_kind`` + ``relation_type`` 등록.
    4. 코드가 허용 집합 안이면 ``relation_kind`` 조회 후 ``relation_type`` 을 **inactive** 로 보장
       (LLM 이 제안한 번들은 active 로 올리지 않음).

``source_media_item_id`` 는 API 일관성을 위해 유지하며, 현재 본문에서는 사용하지 않는다.
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection

from src.relations.relation_type_catalog import (
    ensure_inactive_relation_type_row,
    ensure_relation_kind_for_llm_proposal,
    ensure_relation_subtopic_bundle,
    fetch_relation_kind_id,
)
from src.relations.schema import (
    coerce_topic_fields_mvp,
    description_ko_from_type_name_ko,
    normalize_relation_type_code,
    sanitize_llm_proposed_type_code,
    type_label_from_kind_code,
)


def sync_relation_catalog_from_llm_edges(
    conn: Connection[Any],
    *,
    source_media_item_id: int,
    edges: list[dict[str, Any]],
    llm_prompt_type_codes: frozenset[str],
) -> tuple[int, int]:
    """
    엣지별로 카탈로그만 갱신한다.

    Returns:
        (catalog_synced, catalog_skipped)
        - **synced**: 카탈로그(항상 ``relation_type`` 은 inactive) 갱신이 끝난 엣지 수.
        - **skipped**: 코드 없음·비허용+sanitize 실패·DB에 kind 없음 등.
    """
    _ = source_media_item_id
    synced = 0
    skipped = 0

    for edge in edges:
        topic_ko, subtopic_ko, topic_en, subtopic_en, _topic_auto = coerce_topic_fields_mvp(edge)
        relation_subtopic_id = ensure_relation_subtopic_bundle(
            conn,
            topic_ko=topic_ko,
            subtopic_ko=subtopic_ko,
            topic_en=topic_en,
            subtopic_en=subtopic_en,
        )

        canon = normalize_relation_type_code(edge.get("relation_type_code"))
        if canon is None:
            skipped += 1
            continue

        if canon not in llm_prompt_type_codes:
            safe = sanitize_llm_proposed_type_code(canon)
            if safe is None:
                skipped += 1
                continue
            reason_txt = str(edge.get("reason") or "").strip()
            type_label = type_label_from_kind_code(safe)
            desc_body = description_ko_from_type_name_ko(type_label)
            if reason_txt:
                desc_body = f"{desc_body}\n\n[LLM 근거]\n{reason_txt[:4000]}"
            kid_safe = ensure_relation_kind_for_llm_proposal(
                conn,
                kind_code=safe,
                kind_name_ko=type_label,
                description=desc_body,
                status="inactive",
            )
            ensure_inactive_relation_type_row(
                conn,
                relation_kind_id=kid_safe,
                relation_subtopic_id=relation_subtopic_id,
            )
            synced += 1
            continue

        relation_kind_id = fetch_relation_kind_id(conn, kind_code=canon)
        if relation_kind_id is None:
            skipped += 1
            continue

        ensure_inactive_relation_type_row(
            conn,
            relation_kind_id=relation_kind_id,
            relation_subtopic_id=relation_subtopic_id,
        )
        synced += 1

    return synced, skipped
