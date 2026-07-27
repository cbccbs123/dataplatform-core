"""LLM 엣지에서 **신규 relation_kind** 만 검토용 inactive 로 등록한다.

C+ 슬림화 이후 주제는 엣지 jsonb 라 relation_type/subtopic 동기화가 사라졌다.
active kind 는 그대로 graph_persist 가 엣지로 만들고, 미지의 코드만 여기서 inactive 등록(검토 대기).

설계 불변식
    - active kind 를 여기서 건드리지 않는다 — 활성 어휘 집합은 HITL review.promote_relation_kind 경로로만 확장.
    - 이 모듈 호출 후 graph_persist.sync_graph_edges 가 이어서 엣지를 확정한다(파이프라인 순서 고정).
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
    """LLM이 쓴 관계 코드 중 **아직 통제 어휘에 없는 것**만 검토 대기(inactive)로 등록한다.

    **DB에 쓴다**(신규 코드가 있을 때만). 호출자의 트랜잭션 안에서 돈다. 여기서 등록된 코드는
    같은 사이클에서 엣지가 되지 못하고, 사람이 승인한 다음 주기부터 쓰인다.

    Args:
        edges: 정규화된 LLM 제안 엣지 목록. 각 항목의 ``relation_type_code``·``reason`` 만 본다.
        active_kind_codes: 호출 전에 DB에서 읽어온 활성 kind_code 집합.
                           이 집합에 속하는 코드는 이미 통제어휘에 있으므로 건너뛴다.

    Returns:
        ``(registered, skipped)`` — registered 는 **이번에 새로 등록한 코드 수**(배치 내 중복은
        1회만 센다), skipped 는 이미 있거나 형식 검사에서 버린 수.

    단일 배치 내 중복 코드 처리
        ``seen`` 집합으로 같은 배치 내 중복 코드를 걸러낸다.
        ``ensure_relation_kind_for_llm_proposal`` 이 ON CONFLICT DO UPDATE 라 멱등이긴 하나,
        DB 왕복을 최소화하고 registered 카운트가 실제 신규 등록 수만 반영하게 한다.

    LLM 근거 보존
        ``reason`` 필드를 description 에 첨부해 두면 검토자(HITL)가 승격/반려 판단 시 맥락을 볼 수 있다.
        4000자 상한은 description 컬럼 크기 초과 방지용이다.
    """
    registered = 0
    skipped = 0
    seen: set[str] = set()
    for edge in edges:
        canon = normalize_relation_type_code(edge.get("relation_type_code"))
        # active kind 이거나 이미 이 배치에서 처리한 코드는 DB 재등록 불필요
        if canon is None or canon in active_kind_codes or canon in seen:
            skipped += 1
            continue
        # 형식·레거시 검사: 통과 실패 시 LLM 환각·도메인 혼용 코드로 간주하고 버림
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
