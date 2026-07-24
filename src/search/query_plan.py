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
    """입력 q·mode → 정책(auto|keyword). 자동 tag 승격 없음."""
    mode_norm: SearchMode = "keyword" if mode == "keyword" else "auto"
    return SearchPolicy(content_query=q, mode=mode_norm)


def build_query_plan(q: str, mode: str = "auto") -> QueryPlan:
    """QueryPlan 완성(044 FR-301)."""
    return QueryPlan(policy=build_search_policy(q, mode=mode))


def search_plan_to_meta(plan: QueryPlan) -> dict[str, Any]:
    """포탈 ``meta.search_plan`` minimal 노출(FR-303) — content_query·mode."""
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
