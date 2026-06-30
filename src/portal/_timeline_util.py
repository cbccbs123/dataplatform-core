"""013 — 타임라인 멀티시리즈 피벗 공용 헬퍼. 읽기 전용·결정적(헌법 3조)·LLM 0.

계보(lineage_query)·접근이력(access_log) 타임라인이 동일한 멀티시리즈 형태를 쓰므로,
(key, bucket, count) 행을 series 로 접는 피벗 로직을 한 곳에 둔다(DRY·동기화 부채 제거).
"""
from __future__ import annotations

from typing import Any


def pivot_series(grouped_rows: list[tuple]) -> list[dict]:
    """(key, bucket, count) 행(key ASC·bucket ASC 정렬 가정)을 멀티시리즈로 피벗(순서 보존·결정적).

    DB 가 이미 key ASC·bucket ASC 로 정렬해 주므로, dict 삽입순(3.7+ 보장)이 곧 시리즈 순서다.
    """
    series_map: dict[Any, list] = {}
    order: list = []
    for key, bkt, count in grouped_rows:
        if key not in series_map:
            series_map[key] = []
            order.append(key)
        series_map[key].append(
            {"bucket": bkt.isoformat() if bkt is not None else None, "count": int(count)})
    return [{"key": k, "buckets": series_map[k]} for k in order]
