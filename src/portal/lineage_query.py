"""013 US2 — 자산 계보(asset_lineage) 조회. 읽기 전용·결정적(헌법 3조)·LLM 0.

기록(record_lineage)은 수집·관계 파이프라인이 이미 함. 본 모듈은 활동을 시간순으로 끌어올린다.
**의료(PHI) 제외**: asset 조인으로 domain_label='medical' 자산의 계보는 노출하지 않는다
(검색·상세·대시보드와 일관·헌법 10조·FR-014). 비의료 자산은 status 무관 전부 포함(운영상 failed 계보 필요).
"""
from __future__ import annotations

from typing import Any

# 의료 제외 고정 절(사용자 입력 아님·인젝션 안전). al=asset_lineage, a=asset.
_NONMEDICAL = (
    "FROM asset_lineage al JOIN asset a ON a.asset_id = al.asset_id "
    "WHERE a.domain_label <> 'medical'"
)


def query_asset_lineage(conn: Any, asset_id: str, *, limit: int = 500) -> list[dict]:
    """자산의 활동을 발생 시각순으로 반환(의료 제외) — [{activity, agent, used, generated, occurred_at}]."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT al.activity, al.agent, al.used, al.generated, al.occurred_at "
            + _NONMEDICAL + " AND al.asset_id = %s "
            "ORDER BY al.occurred_at ASC, al.lineage_id ASC LIMIT %s",
            (asset_id, limit))
        return [
            {"activity": act, "agent": ag, "used": used, "generated": gen,
             "occurred_at": ts.isoformat() if ts is not None else None}
            for act, ag, used, gen, ts in cur.fetchall()]


def query_lineage_feed(
    conn: Any, *, since: Any = None, until: Any = None,
    activity: str | None = None, limit: int = 50, offset: int = 0,
) -> dict[str, Any]:
    """기간 내 전 자산 계보 피드(의료 제외·occurred_at DESC, lineage_id DESC·페이징·FR-009b)."""
    conds: list[str] = []
    params: list[Any] = []
    if since is not None:
        conds.append("al.occurred_at >= %s")
        params.append(since)
    if until is not None:
        conds.append("al.occurred_at < %s")
        params.append(until)
    if activity:
        conds.append("al.activity = %s")
        params.append(activity)
    extra = (" AND " + " AND ".join(conds)) if conds else ""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) " + _NONMEDICAL + extra, params)
        total = int(cur.fetchone()[0])
        cur.execute(
            "SELECT al.lineage_id, al.asset_id, al.activity, al.agent, al.occurred_at "
            + _NONMEDICAL + extra
            + " ORDER BY al.occurred_at DESC, al.lineage_id DESC LIMIT %s OFFSET %s",
            [*params, limit, offset])
        rows = [
            {"lineage_id": str(lid), "asset_id": str(aid), "activity": act, "agent": ag,
             "occurred_at": ts.isoformat() if ts is not None else None}
            for lid, aid, act, ag, ts in cur.fetchall()]
    return {"rows": rows, "total": total}
