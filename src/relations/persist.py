"""LLM 엣지에서 **신규 relation_kind** 만 검토용 inactive 로 등록한다.

C+ 슬림화 이후 주제는 엣지 jsonb 라 relation_type/subtopic 동기화가 사라졌다.
active kind 는 그대로 graph_persist 가 엣지로 만들고, 미지의 코드만 여기서 inactive 등록(검토 대기).
"""
from __future__ import annotations

from typing import Any

from psycopg import Connection

from src.relations.relation_type_catalog import ensure_relation_kind_for_llm_proposal
from src.relations.schema import (
    description_ko_from_type_name_ko,
    normalize_relation_type_code,
    sanitize_llm_proposed_type_code,
    type_label_from_kind_code,
)


def register_new_relation_kinds(
    conn: Connection[Any],
    *,
    edges: list[dict[str, Any]],
    active_kind_codes: frozenset[str],
) -> tuple[int, int]:
    """엣지 코드 중 active 집합 밖·형식 통과한 것만 relation_kind 에 inactive 등록. Returns (registered, skipped)."""
    registered = 0
    skipped = 0
    seen: set[str] = set()
    for edge in edges:
        canon = normalize_relation_type_code(edge.get("relation_type_code"))
        if canon is None or canon in active_kind_codes or canon in seen:
            skipped += 1
            continue
        safe = sanitize_llm_proposed_type_code(canon)
        if safe is None:
            skipped += 1
            continue
        label = type_label_from_kind_code(safe)
        desc = description_ko_from_type_name_ko(label)
        reason_txt = str(edge.get("reason") or "").strip()
        if reason_txt:
            desc = f"{desc}\n\n[LLM 근거]\n{reason_txt[:4000]}"
        ensure_relation_kind_for_llm_proposal(
            conn, kind_code=safe, kind_name_ko=label, description=desc, status="inactive")
        seen.add(safe)
        registered += 1
    return registered, skipped
