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
    """확장자를 비교 가능한 형태로 정규화한다 — 유니코드 정규화·소문자화·앞 ``.`` 제거.

    Args:
        value: 사용자가 준 확장자(``.JPG``·``jpg`` 등 표기가 제각각).

    Returns:
        정규화된 확장자(``jpg``).
    """
    return unicodedata.normalize("NFKC", value.strip()).casefold().lstrip(".")


def _parse_date_param(raw: str) -> date | datetime:
    """ISO 문자열을 날짜 또는 일시로 파싱한다.

    Args:
        raw: ``YYYY-MM-DD``(10자) 또는 ISO 8601 일시. 끝의 ``Z`` 는 ``+00:00`` 으로 바꿔 받는다.

    Returns:
        10자면 ``date``, 그 외는 ``datetime``.

    Raises:
        ValueError: ISO 형식이 아닐 때(``fromisoformat`` 이 던진다). 호출자(API)가 400 으로 변환한다.
    """
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
    """API 질의 파라미터를 ``SearchFilters`` 로 파싱한다(순수).

    Args:
        file_ext: 확장자 목록(반복 파라미터). 정규화·중복 제거·정렬해 담는다.
        created_from: 생성일 시작(ISO 문자열). 빈 문자열은 미지정으로 본다.
        created_to: 생성일 끝(ISO 문자열).
        topic: 주제 정확 일치 필터. 색인된 keyword 원문과 맞춰야 하므로 **소문자화하지 않고**
            앞뒤 공백만 자른다.
        subtopic: 세부주제 정확 일치 필터(같은 규칙).

    Returns:
        ``SearchFilters``. **하나도 지정되지 않았으면 ``None``** — 호출부가 "필터 없음"을
        빈 객체와 구분해 처리한다.

    Raises:
        ValueError: 날짜 문자열이 ISO 형식이 아닐 때.
    """
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
    """OpenSearch date 필드에 넣을 ISO 날짜 문자열로 바꾼다(일 단위).

    Args:
        value: 날짜 또는 일시. 일시면 **시각을 버리고 날짜만** 쓴다.

    Returns:
        ``YYYY-MM-DD``.
    """
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def filters_to_opensearch_bool(filters: SearchFilters | None) -> list[dict[str, Any]]:
    """``SearchFilters`` 를 OpenSearch ``bool.filter`` 절 목록으로 바꾼다(BM25·kNN 공통).

    filter 절은 **점수에 기여하지 않는다** — 걸러내기만 하므로 랭킹이 흔들리지 않는다.

    Args:
        filters: 파싱된 필터. ``None`` 이면 빈 목록을 돌려준다(필터 없음).

    Returns:
        절 dict 목록. 지정된 항목만 들어가며, 아무것도 없으면 빈 목록.
    """
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
