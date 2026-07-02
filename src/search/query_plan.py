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
    seed: tuple[str, ...] | None = None,
) -> bool:
    """단일 일반어 seed + 1토큰 휴리스틱(공백 없음·len≤12)."""
    effective = seed if seed is not None else resolve_generic_term_seed()
    token = q.strip()
    if not token or " " in token:
        return False
    if len(token) > 12:
        return False
    norm = _normalize_token(token)
    return norm in {_normalize_token(s) for s in effective}


def merge_generic_term_seed(
    base: tuple[str, ...],
    extra: tuple[str, ...],
) -> tuple[str, ...]:
    """core seed + env extra — NFKC+casefold 중복 제거(결정적·base 우선)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in (*base, *extra):
        norm = _normalize_token(raw)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(raw.strip())
    return tuple(out)


def resolve_generic_term_seed() -> tuple[str, ...]:
    """settings 초기화 시 merge seed, 미초기화면 core 6개만."""
    try:
        from src.config.settings import get_current_settings

        return get_current_settings().search_generic_term_seed
    except RuntimeError:
        return sc.GENERIC_SINGLE_TERM_SEED


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


# 057 FR-203: mode=keyword 안내는 **API 파라미터 조작 지시**라 dev-facing(프론트가 정규식으로 걸러온 문구).
# 생성 출처가 문구를 상수로 소유하고, 아래 audience 매핑이 그 문구의 대상(dev)을 권위 있게 태깅한다 —
# 프론트 정규식(isDevFacingSuggestion) 표류를 서버 단일 판정으로 대체(spec FR-203·B4).
_KEYWORD_MODE_SUGGESTION = "단어 포함 문서를 보려면 mode=keyword 로 검색하세요."
# suggestion 텍스트 → audience("user"|"dev"). 매핑에 없는 문구는 보수적으로 "user"(표시) —
# 현행 프론트가 dev 만 거르고 나머지는 사용자에게 노출하던 동작과 정합(기본값 = 사용자 표시).
_SUGGESTION_AUDIENCE: dict[str, str] = {_KEYWORD_MODE_SUGGESTION: "dev"}


def _suggestion_audience(text: str) -> str:
    """suggestion 문구의 대상 청중을 반환(미매핑 → 'user' 보수 폴백·프론트 노출 규칙 정합)."""
    return _SUGGESTION_AUDIENCE.get(text, "user")


def _build_suggestions(policy: SearchPolicy) -> tuple[str, ...]:
    """generic single term + auto → keyword 모드 안내(FR-302·spec C4)."""
    if policy.generic_single_term and policy.mode == "auto":
        return (_KEYWORD_MODE_SUGGESTION,)
    return ()


def build_query_plan(q: str, mode: str = "auto") -> QueryPlan:
    """QueryPlan 완성 — G3 suggestions 포함."""
    policy = build_search_policy(q, mode=mode)
    return QueryPlan(policy=policy, suggestions=_build_suggestions(policy))


def search_plan_to_meta(plan: QueryPlan) -> dict[str, Any]:
    """포탈 ``meta.search_plan`` minimal 노출(FR-303).

    057 FR-203: ``suggestions`` 를 ``[{text, audience}]`` 로 태깅해 내려, 프론트가 dev 힌트를
    정규식으로 분류하던 로직을 제거한다(audience 로 필터). 제안이 없으면 키 자체를 두지 않는다
    (기존 minimal 규칙 보존). 이는 P2 응답 shape 변경이라 프론트 lockstep 이 필요하다(spec C2).
    """
    p = plan.policy
    out: dict[str, Any] = {
        "content_query": p.content_query,
        "lexical_rescue": p.lexical_rescue,
        "generic_single_term": p.generic_single_term,
        "mode": p.mode,
    }
    if plan.suggestions:
        out["suggestions"] = [
            {"text": t, "audience": _suggestion_audience(t)} for t in plan.suggestions
        ]
    return out


__all__ = [
    "LexicalRescueMode",
    "QueryPlan",
    "SearchMode",
    "SearchPolicy",
    "build_query_plan",
    "build_search_policy",
    "is_generic_single_term",
    "merge_generic_term_seed",
    "resolve_generic_term_seed",
    "search_plan_to_meta",
]
