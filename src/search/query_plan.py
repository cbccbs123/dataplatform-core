"""044 G2/G3 — QueryPlan·SearchPolicy 얇은 계층(LLM 0 · 결정적)."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from src.config import search_constants as sc

LexicalRescueMode = Literal["normal", "restricted"]
SearchMode = Literal["auto", "keyword"]


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    """검색 rescue·질의 정책(044 FR-301 subset — G2 stub)."""

    content_query: str
    lexical_rescue: LexicalRescueMode
    generic_single_term: bool
    mode: SearchMode


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """044 FR-301 — policy + UI suggestion(자동 필터 승격 없음)."""

    policy: SearchPolicy
    suggestions: tuple[str, ...] = ()


def _normalize_token(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip()).casefold()


def is_generic_single_term(
    q: str,
    *,
    seed: tuple[str, ...] = sc.GENERIC_SINGLE_TERM_SEED,
) -> bool:
    """단일 일반어 seed + 1토큰 휴리스틱(공백 없음·len≤12)."""
    token = q.strip()
    if not token or " " in token:
        return False
    if len(token) > 12:
        return False
    norm = _normalize_token(token)
    return norm in {_normalize_token(s) for s in seed}


def build_search_policy(q: str, mode: str = "auto") -> SearchPolicy:
    """입력 q·mode → rescue 정책(자동 tag 승격 없음)."""
    mode_norm: SearchMode = "keyword" if mode == "keyword" else "auto"
    generic = is_generic_single_term(q)
    rescue: LexicalRescueMode = "restricted" if generic and mode_norm == "auto" else "normal"
    return SearchPolicy(
        content_query=q,
        lexical_rescue=rescue,
        generic_single_term=generic,
        mode=mode_norm,
    )


def _build_suggestions(policy: SearchPolicy) -> tuple[str, ...]:
    """generic single term + auto → keyword 모드 안내(FR-302·spec C4)."""
    if policy.generic_single_term and policy.mode == "auto":
        return (
            "단어 포함 문서를 보려면 mode=keyword 로 검색하세요.",
        )
    return ()


def build_query_plan(q: str, mode: str = "auto") -> QueryPlan:
    """QueryPlan 완성 — G3 suggestions 포함."""
    policy = build_search_policy(q, mode=mode)
    return QueryPlan(policy=policy, suggestions=_build_suggestions(policy))


def search_plan_to_meta(plan: QueryPlan) -> dict[str, Any]:
    """포탈 ``meta.search_plan`` minimal 노출(FR-303)."""
    p = plan.policy
    out: dict[str, Any] = {
        "content_query": p.content_query,
        "lexical_rescue": p.lexical_rescue,
        "generic_single_term": p.generic_single_term,
        "mode": p.mode,
    }
    if plan.suggestions:
        out["suggestions"] = list(plan.suggestions)
    return out


__all__ = [
    "LexicalRescueMode",
    "QueryPlan",
    "SearchMode",
    "SearchPolicy",
    "build_query_plan",
    "build_search_policy",
    "is_generic_single_term",
    "search_plan_to_meta",
]
