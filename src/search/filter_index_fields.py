"""045 Phase B — OS 색인·필터용 파생 필드(순수·결정적)."""

from __future__ import annotations

import os
import unicodedata
from datetime import date, datetime
from typing import Any


def derive_file_ext(fs_path: str) -> str | None:
    """``fs_path`` basename 마지막 확장자 — lowercase NFKC. 없으면 None."""
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
    """``asset.created_at`` → OS date 문자열(YYYY-MM-DD, UTC 일 단위)."""
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
    """``asset_to_doc`` 에 병합할 filter_kw·filter_date 부분 dict."""
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
