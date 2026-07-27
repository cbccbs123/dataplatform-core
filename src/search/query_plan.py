"""044 G2/G3 — QueryPlan·SearchPolicy 얇은 계층(LLM 0 · 결정적).

2026-07-24 mode 슬림: generic_single_term·restricted rescue·keyword 안내(suggestion)·seed 제거.
``mode``(auto|keyword)만 유지하며, 그 효과는 ``query_evidence.lexical_rescue_keep`` 의 게이트-실패
rescue 임계에만 반영된다(keyword=관대한 EVIDENCE_KEYWORD_THRESHOLD / auto=EVIDENCE_NORMAL_THRESHOLD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SearchMode = Literal["auto", "keyword"]


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    """검색 정책(044 FR-301 subset). ``mode`` 만 rescue 임계에 반영(query_evidence)."""

    content_query: str
    mode: SearchMode


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """044 FR-301 — policy(자동 필터 승격 없음)."""

    policy: SearchPolicy


def build_search_policy(q: str, mode: str = "auto") -> SearchPolicy:
    """질의와 모드로 검색 정책을 만든다(순수·LLM 0).

    Args:
        q: 사용자 질의 원문.
        mode: ``keyword`` 면 키워드 모드, **그 밖의 값은 모두** ``auto``(알 수 없는 값을 예외로
            올리지 않고 안전한 기본으로 접는다).

    Returns:
        불변 ``SearchPolicy``. ``mode`` 는 뒤쪽 rescue 임계에만 영향을 준다 — 키워드 모드가 더
        관대해 어휘가 맞은 행을 잘 살린다.
    """
    mode_norm: SearchMode = "keyword" if mode == "keyword" else "auto"
    return SearchPolicy(content_query=q, mode=mode_norm)


def build_query_plan(q: str, mode: str = "auto") -> QueryPlan:
    """정책을 감싼 ``QueryPlan`` 을 만든다.

    Args:
        q: 사용자 질의 원문.
        mode: ``auto``|``keyword``.

    Returns:
        ``QueryPlan``(현재는 policy 한 필드만 갖는 얇은 래퍼).
    """
    return QueryPlan(policy=build_search_policy(q, mode=mode))


def search_plan_to_meta(plan: QueryPlan) -> dict[str, Any]:
    """계획을 API 응답 ``meta.search_plan`` 에 실을 최소 dict 로 바꾼다.

    Args:
        plan: ``build_query_plan`` 결과.

    Returns:
        ``{content_query, mode}`` — 내부 구조를 그대로 노출하지 않고 두 값만 공개한다.
    """
    p = plan.policy
    return {"content_query": p.content_query, "mode": p.mode}


__all__ = [
    "QueryPlan",
    "SearchMode",
    "SearchPolicy",
    "build_query_plan",
    "build_search_policy",
    "search_plan_to_meta",
]
