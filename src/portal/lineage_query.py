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


_FEED_COLS = "lineage_id, asset_id, activity, agent, occurred_at"


def query_lineage_feed(conn: Any, *, since: Any = None, until: Any = None,
                       activity: str | None = None, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """기간 내 전 자산 계보 피드(occurred_at DESC, lineage_id DESC tiebreak·페이징·FR-009b)."""
    conds: list[str] = []
    params: list[Any] = []
    if since is not None:
        conds.append("occurred_at >= %s")
        params.append(since)
    if until is not None:
        conds.append("occurred_at < %s")
        params.append(until)
    if activity:
        conds.append("activity = %s")
        params.append(activity)
    clause = (" WHERE " + " AND ".join(conds)) if conds else ""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM asset_lineage" + clause, params)
        total = int(cur.fetchone()[0])
        cur.execute(
            f"SELECT {_FEED_COLS} FROM asset_lineage{clause} "
            "ORDER BY occurred_at DESC, lineage_id DESC LIMIT %s OFFSET %s",
            [*params, limit, offset])
        rows = [
            {"lineage_id": str(lid), "asset_id": str(aid), "activity": act, "agent": ag,
             "occurred_at": ts.isoformat() if ts is not None else None}
            for lid, aid, act, ag, ts in cur.fetchall()]
    return {"rows": rows, "total": total}
