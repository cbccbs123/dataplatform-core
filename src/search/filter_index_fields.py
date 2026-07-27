"""045 Phase B — OS 색인·필터용 파생 필드(순수·결정적)."""

from __future__ import annotations

import os
import unicodedata
from datetime import date, datetime
from typing import Any


def derive_file_ext(fs_path: str) -> str | None:
    """경로에서 확장자를 뽑아 정규화한다(소문자·유니코드 NFKC).

    Args:
        fs_path: 자산 경로. 빈 값도 받는다.

    Returns:
        확장자(``jpg``). 점이 없거나 점으로 끝나면 ``None``.
    """
    base = os.path.basename(fs_path or "")
    if "." not in base or base.endswith("."):
        return None
    ext = base.rsplit(".", 1)[-1].strip()
    if not ext:
        return None
    return unicodedata.normalize("NFKC", ext).casefold()


def derive_source_dataset(fs_path: str) -> str:
    """경로 규칙으로 source_dataset 키를 반환한다(ADR 2026-06-24-search-filter-v1).

    규칙 순서: ``sample_data/data1|2|3`` 경로면 그 키 → 아니면 경로에 ``wikipedia``/``youtube`` 가 있으면
    그 출처 → 그 외 ``unknown``. 현행 코퍼스는 새 경로 ``data_all`` 이라 data1/2/3 규칙은 불발하고,
    파일명·경로의 ``youtube``/``wikipedia`` substring 폴백으로 분류된다(그 외 파일은 unknown 패싯).

    ℹ️ 이 값은 **색인에만 남는다** — 검색 필터로는 노출하지 않는다(경로 휴리스틱이라 약 70%가
    ``unknown`` 이라 패싯으로 쓸모가 없었다). 출처 추적용 메타로만 존치한다.

    Args:
        fs_path: 자산 경로.

    Returns:
        출처 키(``data1``|``data2``|``data3``|``wikipedia``|``youtube``|``unknown``).
    """
    p = unicodedata.normalize("NFKC", fs_path or "").casefold().replace("\\", "/")
    for n in ("data1", "data2", "data3"):
        if f"/sample_data/{n}/" in p or f"sample_data/{n}/" in p:
            return n
    if "wikipedia" in p:
        return "wikipedia"
    if "youtube" in p:
        return "youtube"
    return "unknown"


def _coerce_filter_date(value: Any) -> str | None:
    """생성 시각을 OpenSearch date 필드용 문자열로 바꾼다(일 단위).

    Args:
        value: ``datetime``·``date``·PostgreSQL timestamptz 문자열·``None`` 무엇이든 받는다.
            문자열은 앞 10자만 취한다(시각·시간대는 버린다).

    Returns:
        ``YYYY-MM-DD``. 값이 없거나 알 수 없는 타입이면 ``None``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value.strip():
        # PG timestamptz 문자열 — 앞 10자 date
        return value.strip()[:10]
    return None


def build_filter_index_fields(
    *,
    fs_path: str,
    created_at: Any = None,
) -> dict[str, Any]:
    """색인 문서에 병합할 필터 전용 필드를 만든다(순수).

    Args:
        fs_path: 자산 경로(확장자·출처를 여기서 파생한다).
        created_at: 생성 시각. ``None`` 이면 ``filter_date`` 를 아예 넣지 않는다.

    Returns:
        ``{"filter_kw": {...}, "filter_date": {...}}`` 부분 dict. **값이 없는 키는 넣지 않는다** —
        빈 문자열로 채우면 색인에 쓸모없는 항목이 생기기 때문이다.
    """
    out: dict[str, Any] = {}
    filter_kw: dict[str, str] = {}
    ext = derive_file_ext(fs_path)
    if ext:
        filter_kw["file_ext"] = ext
    filter_kw["source_dataset"] = derive_source_dataset(fs_path)
    if filter_kw:
        out["filter_kw"] = filter_kw
    created = _coerce_filter_date(created_at)
    if created:
        out["filter_date"] = {"created_at": created}
    return out


__all__ = [
    "build_filter_index_fields",
    "derive_file_ext",
    "derive_source_dataset",
]
