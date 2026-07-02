"""013 US2 — 자산 계보(asset_lineage) 조회. 읽기 전용·결정적(헌법 3조)·LLM 0.

기록(record_lineage)은 수집·관계 파이프라인이 이미 함. 본 모듈은 활동을 시간순으로 끌어올린다.
**의료(PHI) 제외**: asset 조인으로 domain_label='medical' 자산의 계보는 노출하지 않는다
(검색·상세·대시보드와 일관·헌법 10조·FR-014). 비의료 자산은 status 무관 전부 포함(운영상 failed 계보 필요).
"""
from __future__ import annotations

from typing import Any

from src.portal._timeline_util import pivot_series

# 의료 제외 고정 절(사용자 입력 아님·인젝션 안전). al=asset_lineage, a=asset.
_NONMEDICAL = (
    "FROM asset_lineage al JOIN asset a ON a.asset_id = al.asset_id "
    "WHERE a.domain_label <> 'medical'"
)
# 파일 확장자(file_ext) = a.fs_path 마지막 .세그먼트(소문자). 고정 SQL·raw 정규식(인젝션 안전).
_EXT_EXPR = r"lower(substring(a.fs_path from '\.([^./]+)$'))"


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
    conn: Any, *, since: Any = None, until: Any = None, activity: str | None = None,
    modality: str | None = None, status: str | None = None, file_ext: str | None = None,
    limit: int = 50, offset: int = 0,
) -> dict[str, Any]:
    """기간 내 전 자산 계보 피드(의료 제외·occurred_at DESC, lineage_id DESC·페이징·FR-009b).

    필터: 기간(since/until)·활동(activity)·**자산 차원**(modality·status·file_ext — asset 조인).
    대시보드 슬라이스용.
    """
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
    if modality:
        conds.append("a.modality = %s")
        params.append(modality)
    if status:
        conds.append("a.status = %s")
        params.append(status)
    if file_ext:
        conds.append(f"{_EXT_EXPR} = %s")
        params.append(file_ext)
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
    # FR-701(054): 페이징 봉투 통일({rows,total,limit,offset}) — 프론트 목록 페이지/맨앞·맨끝 이동.
    return {"rows": rows, "total": total, "limit": limit, "offset": offset}


_INTERVALS = {"day", "hour", "month"}  # date_trunc 화이트리스트(f-string 인젝션 방지·054 month 추가)
# 멀티시리즈 group_by 화이트리스트 → 컬럼식(고정 매핑·사용자 입력은 키로만 조회·인젝션 안전).
_GROUP_COLS = {"activity": "al.activity", "modality": "a.modality", "status": "a.status"}


def _lineage_filter(since: Any, until: Any, activity: str | None) -> tuple[str, list[Any]]:
    """계보 공통 필터 절(기간·활동) + 파라미터. _NONMEDICAL 뒤에 AND 로 붙인다."""
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
    return ((" AND " + " AND ".join(conds)) if conds else ""), params


def lineage_stats(conn: Any, *, since: Any = None, until: Any = None,
                  activity: str | None = None) -> dict[str, Any]:
    """계보 집계(차트·KPI용·FR-009g 보완) — 총계 + 활동별·일별·modality·status·file_ext별. 의료 제외·결정적."""
    extra, params = _lineage_filter(since, until, activity)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) " + _NONMEDICAL + extra, params)
        total = int(cur.fetchone()[0])
        cur.execute("SELECT al.activity, COUNT(*) " + _NONMEDICAL + extra
                    + " GROUP BY al.activity ORDER BY COUNT(*) DESC, al.activity ASC", params)
        by_activity = [{"activity": a, "count": int(c)} for a, c in cur.fetchall()]
        cur.execute("SELECT al.occurred_at::date AS d, COUNT(*) " + _NONMEDICAL + extra
                    + " GROUP BY d ORDER BY d ASC", params)
        by_day = [{"day": d.isoformat() if d is not None else None, "count": int(c)}
                  for d, c in cur.fetchall()]
        cur.execute("SELECT a.modality, COUNT(*) " + _NONMEDICAL + extra
                    + " GROUP BY a.modality ORDER BY COUNT(*) DESC, a.modality ASC", params)
        by_modality = [{"modality": m, "count": int(c)} for m, c in cur.fetchall()]
        cur.execute("SELECT a.status, COUNT(*) " + _NONMEDICAL + extra
                    + " GROUP BY a.status ORDER BY COUNT(*) DESC, a.status ASC", params)
        by_status = [{"status": s, "count": int(c)} for s, c in cur.fetchall()]
        cur.execute(f"SELECT {_EXT_EXPR} AS ext, COUNT(*) " + _NONMEDICAL + extra
                    + " GROUP BY ext ORDER BY COUNT(*) DESC, ext ASC NULLS LAST", params)
        by_file_ext = [{"file_ext": e, "count": int(c)} for e, c in cur.fetchall()]
    return {"total": total, "by_activity": by_activity, "by_day": by_day,
            "by_modality": by_modality, "by_status": by_status, "by_file_ext": by_file_ext}


def lineage_timeline(conn: Any, *, since: Any = None, until: Any = None, activity: str | None = None,
                     interval: str = "day", group_by: str | None = None) -> dict[str, Any]:
    """계보 시계열(차트용·access timeline 과 대칭). group_by(activity/modality/status) 주면 멀티시리즈.

    의료 제외·결정적(시리즈 key ASC·버킷 ASC). group_by 미지정이면 단일 시리즈({interval, buckets}).
    """
    trunc = interval if interval in _INTERVALS else "day"
    extra, params = _lineage_filter(since, until, activity)
    with conn.cursor() as cur:
        if group_by in _GROUP_COLS:
            gcol = _GROUP_COLS[group_by]
            cur.execute(
                f"SELECT {gcol} AS key, date_trunc('{trunc}', al.occurred_at) AS bkt, COUNT(*) "
                + _NONMEDICAL + extra + " GROUP BY key, bkt ORDER BY key ASC, bkt ASC", params)
            return {"interval": trunc, "group_by": group_by, "series": pivot_series(cur.fetchall())}
        cur.execute(f"SELECT date_trunc('{trunc}', al.occurred_at) AS bkt, COUNT(*) "
                    + _NONMEDICAL + extra + " GROUP BY bkt ORDER BY bkt ASC", params)
        buckets = [{"bucket": b.isoformat() if b is not None else None, "count": int(c)}
                   for b, c in cur.fetchall()]
        return {"interval": trunc, "buckets": buckets}
