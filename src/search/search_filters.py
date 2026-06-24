"""045 Phase B — 검색 선필터(SearchFilters) · OS bool.filter 변환."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
import unicodedata

# v1 allowlist — ADR 2026-06-24-search-filter-v1-decisions.md
FILTERABLE_FILE_EXT = "file_ext"
FILTERABLE_SOURCE_DATASET = "source_dataset"
FILTERABLE_CREATED_FROM = "created_from"
FILTERABLE_CREATED_TO = "created_to"

V1_FILTER_PARAM_NAMES = frozenset({
    FILTERABLE_FILE_EXT,
    FILTERABLE_SOURCE_DATASET,
    FILTERABLE_CREATED_FROM,
    FILTERABLE_CREATED_TO,
})


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """portal/CLI 명시 필터 — 자동 질의 승격 없음(044)."""

    file_exts: tuple[str, ...] = ()
    source_datasets: tuple[str, ...] = ()
    created_from: date | datetime | None = None
    created_to: date | datetime | None = None


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
    source_dataset: list[str] | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
) -> SearchFilters | None:
    """portal repeated query params → SearchFilters. 전부 비어 있으면 None."""
    exts = tuple(
        sorted({_norm_ext(x) for x in (file_ext or []) if x and x.strip()})
    )
    datasets = tuple(
        sorted({x.strip().casefold() for x in (source_dataset or []) if x and x.strip()})
    )
    cf = _parse_date_param(created_from) if created_from and created_from.strip() else None
    ct = _parse_date_param(created_to) if created_to and created_to.strip() else None
    if not exts and not datasets and cf is None and ct is None:
        return None
    return SearchFilters(
        file_exts=exts,
        source_datasets=datasets,
        created_from=cf,
        created_to=ct,
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
    if filters.source_datasets:
        clauses.append({
            "terms": {"filter_kw.source_dataset": sorted(filters.source_datasets)},
        })
    if filters.created_from is not None:
        gte = _to_utc_date(filters.created_from)
        clauses.append({"range": {"filter_date.created_at": {"gte": gte}}})
    if filters.created_to is not None:
        lte = _to_utc_date(filters.created_to)
        clauses.append({"range": {"filter_date.created_at": {"lte": lte}}})
    return clauses


__all__ = [
    "FILTERABLE_CREATED_FROM",
    "FILTERABLE_CREATED_TO",
    "FILTERABLE_FILE_EXT",
    "FILTERABLE_SOURCE_DATASET",
    "SearchFilters",
    "V1_FILTER_PARAM_NAMES",
    "filters_to_opensearch_bool",
    "parse_search_filters",
]
