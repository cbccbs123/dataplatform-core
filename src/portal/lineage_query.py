"""013 US2 — 자산 계보(asset_lineage) 조회. 읽기 전용·결정적(헌법 3조)·LLM 0.

기록(record_lineage)은 수집·관계 파이프라인이 이미 함. 본 모듈은 활동을 시간순으로 끌어올린다.
"""
from __future__ import annotations

from typing import Any

_SELECT = (
    "SELECT activity, agent, used, generated, occurred_at FROM asset_lineage "
    "WHERE asset_id = %s ORDER BY occurred_at ASC, lineage_id ASC LIMIT %s"
)


def query_asset_lineage(conn: Any, asset_id: str, *, limit: int = 500) -> list[dict]:
    """자산의 활동을 발생 시각순으로 반환 — [{activity, agent, used, generated, occurred_at}]."""
    with conn.cursor() as cur:
        cur.execute(_SELECT, (asset_id, limit))
        return [
            {"activity": act, "agent": ag, "used": used, "generated": gen,
             "occurred_at": ts.isoformat() if ts is not None else None}
            for act, ag, used, gen, ts in cur.fetchall()]
