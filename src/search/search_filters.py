"""045 Phase B — 검색 선필터(SearchFilters) · OS bool.filter 변환."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """portal/CLI 명시 필터 — 자동 질의 승격 없음(044)."""

    file_exts: tuple[str, ...] = ()
    created_from: date | datetime | None = None
    created_to: date | datetime | None = None
    # 056 FR-503 — 주제/하위주제 keyword terms 필터. 색인된 ``topics``/``subtopics`` (keyword)
    # 원문과 정확 일치해야 하므로 casefold/정규화하지 않는다(strip 만·parse 단계). 단일 값이며
    # terms 절로 변환해 bool.filter 에 넣는다 → **결정적·랭킹 무영향**(topics_text boost 철회).
    topic: str | None = None
    subtopic: str | None = None


def _norm_ext(value: str) -> str:
    return unicodedata.normalize("NFKC", value.strip()).casefold().lstrip(".")


def _parse_date_param(raw: str) -> date | datetime:
    text = raw.strip()
    if len(text) == 10:
        return date.fromisoformat(text)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_search_filters(
    *,
    file_ext: list[str] | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    topic: str | None = None,
    subtopic: str | None = None,
) -> SearchFilters | None:
    """portal repeated query params → SearchFilters. 전부 비어 있으면 None."""
    exts = tuple(
        sorted({_norm_ext(x) for x in (file_ext or []) if x and x.strip()})
    )
    cf = _parse_date_param(created_from) if created_from and created_from.strip() else None
    ct = _parse_date_param(created_to) if created_to and created_to.strip() else None
    # 056: 주제/하위주제는 색인 keyword 원문과 정확 일치용이라 strip 만(casefold·소문자화 금지).
    topic_v = topic.strip() if topic and topic.strip() else None
    subtopic_v = subtopic.strip() if subtopic and subtopic.strip() else None
    if not exts and cf is None and ct is None and topic_v is None and subtopic_v is None:
        return None
    return SearchFilters(
        file_exts=exts,
        created_from=cf,
        created_to=ct,
        topic=topic_v,
        subtopic=subtopic_v,
    )


def _to_utc_date(value: date | datetime) -> str:
    """OS date 필드용 ISO date(UTC 일 단위)."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def filters_to_opensearch_bool(filters: SearchFilters | None) -> list[dict[str, Any]]:
    """SearchFilters → OpenSearch bool.filter 절 리스트(BM25·kNN 공통)."""
    if filters is None:
        return []
    clauses: list[dict[str, Any]] = []
    if filters.file_exts:
        clauses.append({"terms": {"filter_kw.file_ext": sorted(filters.file_exts)}})
    if filters.created_from is not None:
        gte = _to_utc_date(filters.created_from)
        clauses.append({"range": {"filter_date.created_at": {"gte": gte}}})
    if filters.created_to is not None:
        lte = _to_utc_date(filters.created_to)
        clauses.append({"range": {"filter_date.created_at": {"lte": lte}}})
    # 056 FR-503 — 주제/하위주제 terms 필터. 색인된 keyword 필드(top-level ``topics``/``subtopics``·
    # opensearch_sync.build_index_body)에 정확 일치. 단일 값도 terms(1원소 배열)로 두어 향후 다중
    # 확장(반복 파라미터)과 형상 일관. filter 절이라 점수 기여 0 → 랭킹 무영향(결정적).
    if filters.topic:
        clauses.append({"terms": {"topics": [filters.topic]}})
    if filters.subtopic:
        clauses.append({"terms": {"subtopics": [filters.subtopic]}})
    return clauses


__all__ = [
    "SearchFilters",
    "filters_to_opensearch_bool",
    "parse_search_filters",
]
