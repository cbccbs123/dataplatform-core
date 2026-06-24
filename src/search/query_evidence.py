"""044 — 필드 evidence score 순수 함수(헌법 3조 · LLM 0).

``matched_queries``(OpenSearch named query _name) 기반 strong/weak 가중 합산.
임계·가중치 기본값은 ``src.config.search_constants`` 단일 출처.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from src.config import search_constants as sc
from src.search.query_plan import SearchPolicy

# named query _name → weight(044 plan §Evidence weights).
EVIDENCE_WEIGHTS: dict[str, float] = {
    "hit_keywords": sc.EVIDENCE_HIT_KEYWORDS_WEIGHT,
    "hit_labels": sc.EVIDENCE_HIT_LABELS_WEIGHT,
    "hit_file_name": sc.EVIDENCE_HIT_FILE_NAME_WEIGHT,
    "hit_summary": sc.EVIDENCE_HIT_SUMMARY_WEIGHT,
    "hit_search_text": sc.EVIDENCE_HIT_SEARCH_TEXT_WEIGHT,
}

_STRONG_EVIDENCE_NAMES: frozenset[str] = frozenset(
    {"hit_keywords", "hit_labels", "hit_file_name"}
)


def evidence_score(
    matched_queries: Collection[str] | None,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """matched_queries 1/0 × weight 합(순수·결정적).

    unknown _name 은 무시. strong hit 가 있으면 ``hit_search_text`` 가중은 스킵(dedup).
    """
    w = weights or EVIDENCE_WEIGHTS
    names = list(matched_queries or [])
    has_strong = any(n in _STRONG_EVIDENCE_NAMES for n in names if n in w)
    total = 0.0
    for name in names:
        weight = w.get(name)
        if weight is None:
            continue
        if name == "hit_search_text" and has_strong:
            continue
        total += weight
    return total


def strong_evidence_score(
    matched_queries: Collection[str] | None,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """strong tier named query 가중 합만."""
    w = weights or EVIDENCE_WEIGHTS
    total = 0.0
    for name in matched_queries or []:
        if name in _STRONG_EVIDENCE_NAMES and name in w:
            total += w[name]
    return total


def lexical_rescue_keep(
    matched_queries: Collection[str] | None,
    *,
    policy: SearchPolicy,
    rescue_enabled: bool,
    weights: Mapping[str, float] | None = None,
) -> tuple[bool, str]:
    """게이트 실패·BM25 행 1건의 rescue keep/drop(순수·결정적 · 044 FR-202).

    ``rescue_enabled=False`` 이면 legacy(``has_lexical`` 경로) — 항상 keep.
    """
    if not rescue_enabled:
        return True, "legacy_lexical"

    if matched_queries is None:
        # mock·구 hit( matched_queries 키 없음) — 관측 전 호환 keep.
        return True, "legacy_no_matched_queries"

    mq = list(matched_queries)
    if not mq:
        return False, "dropped_no_evidence"

    ev = evidence_score(mq, weights=weights)
    strong = strong_evidence_score(mq, weights=weights)

    if policy.mode == "keyword":
        if ev >= sc.EVIDENCE_KEYWORD_THRESHOLD:
            return True, "evidence_keyword"
        return False, "dropped_weak"

    if policy.lexical_rescue == "restricted":
        if strong >= sc.EVIDENCE_RESTRICTED_STRONG_THRESHOLD:
            return True, "evidence_restricted"
        return False, "dropped_weak"

    if ev >= sc.EVIDENCE_NORMAL_THRESHOLD:
        return True, "evidence_normal"
    return False, "dropped_weak"


__all__ = [
    "EVIDENCE_WEIGHTS",
    "evidence_score",
    "lexical_rescue_keep",
    "strong_evidence_score",
]
