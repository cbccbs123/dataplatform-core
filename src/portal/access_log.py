"""013 US3 — API 접근 이력(access_log) 기록·조회·집계 + 접근 action 도출.

기록은 append-only 감사 write(013 FR-012 감사 무결성). 조회·집계는 읽기 전용·결정적(헌법 3조)·LLM 0.
자산 데이터/스키마는 무변경(헌법 6조) — access_log 만 append-only 로 적재한다.
portal_api 미들웨어가 derive_access_action 으로 (action, asset_id)를 정해 record_access 로 적재한다.
"""
from __future__ import annotations

import json
from typing import Any

from src.database.ids import uuid7

_INSERT = (
    "INSERT INTO access_log (access_id, asset_id, user_id, action, detail) "
    "VALUES (%s, %s, %s, %s, %s::jsonb)"
)
_COLS = "access_id, action, user_id, asset_id, occurred_at"


def record_access(conn: Any, *, action: str, user_id: str,
                  asset_id: str | None = None, detail: dict | None = None) -> str:
    """access_log 한 행 INSERT(append-only). access_id(uuid7) 반환·occurred_at 은 DB now()."""
    access_id = str(uuid7())
    with conn.cursor() as cur:
        cur.execute(_INSERT, (access_id, asset_id, user_id, action,
                              json.dumps(detail or {}, ensure_ascii=False)))
    return access_id


def derive_access_action(method: str, path: str) -> tuple[str, str | None] | None:
    """데이터 접근 GET 라우트 → (action, asset_id). 그 외(감사뷰·비GET·health 등)는 None(기록 안 함)."""
    if method.upper() != "GET":
        return None
    p = path.rstrip("/")
    if p == "/search":
        return ("search", None)
    if p.startswith("/assets/"):
        parts = p[len("/assets/"):].split("/")
        asset_id = parts[0]
        if not asset_id:
            return None
        if len(parts) == 1:
            return ("asset_view", asset_id)
        if len(parts) == 2 and parts[1] == "download":
            return ("download", asset_id)
        if len(parts) == 2 and parts[1] == "bundle":
            return ("bundle", asset_id)
    return None


def _filter_clause(conds: list[str]) -> str:
    return (" WHERE " + " AND ".join(conds)) if conds else ""


def query_access_logs(conn: Any, *, user_id: str | None = None, action: str | None = None,
                      since: Any = None, until: Any = None,
                      limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """필터(사용자·action·기간)·페이징 조회. occurred_at DESC, access_id DESC tiebreak(결정적)."""
    conds: list[str] = []
    params: list[Any] = []
    if user_id:
        conds.append("user_id = %s")
        params.append(user_id)
    if action:
        conds.append("action = %s")
        params.append(action)
    if since is not None:
        conds.append("occurred_at >= %s")
        params.append(since)
    if until is not None:
        conds.append("occurred_at < %s")
        params.append(until)
    clause = _filter_clause(conds)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM access_log" + clause, params)
        total = int(cur.fetchone()[0])
        cur.execute(
            f"SELECT {_COLS} FROM access_log{clause} "
            "ORDER BY occurred_at DESC, access_id DESC LIMIT %s OFFSET %s",
            [*params, limit, offset])
        rows = [
            {"access_id": str(a), "action": act, "user_id": u,
             "asset_id": str(aid) if aid is not None else None,
             "occurred_at": ts.isoformat() if ts is not None else None}
            for a, act, u, aid, ts in cur.fetchall()]
    return {"rows": rows, "total": total}


def access_log_stats(conn: Any, *, since: Any = None, until: Any = None) -> dict[str, Any]:
    """기본 집계: 총계·action별·user별 호출 수(count DESC, key ASC tiebreak·결정적·FR-009a)."""
    conds: list[str] = []
    params: list[Any] = []
    if since is not None:
        conds.append("occurred_at >= %s")
        params.append(since)
    if until is not None:
        conds.append("occurred_at < %s")
        params.append(until)
    clause = _filter_clause(conds)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM access_log" + clause, params)
        total = int(cur.fetchone()[0])
        cur.execute(f"SELECT action, COUNT(*) FROM access_log{clause} "
                    "GROUP BY action ORDER BY COUNT(*) DESC, action ASC", params)
        by_action = [{"action": a, "count": int(c)} for a, c in cur.fetchall()]
        cur.execute(f"SELECT user_id, COUNT(*) FROM access_log{clause} "
                    "GROUP BY user_id ORDER BY COUNT(*) DESC, user_id ASC", params)
        by_user = [{"user_id": u, "count": int(c)} for u, c in cur.fetchall()]
    return {"total": total, "by_action": by_action, "by_user": by_user}


_TIMELINE_INTERVALS = {"day", "hour"}  # date_trunc 화이트리스트(f-string 인젝션 방지)


def access_log_timeline(conn: Any, *, since: Any = None, until: Any = None,
                        action: str | None = None, interval: str = "day") -> dict[str, Any]:
    """시계열 타임라인: 버킷(day/hour)별 호출 수(bucket ASC·결정적·FR-009c). action 필터=api별."""
    # trunc 은 화이트리스트라 f-string 안전, 그 외(since/until/action) 값은 모두 %s 바인딩
    trunc = interval if interval in _TIMELINE_INTERVALS else "day"
    conds: list[str] = []
    params: list[Any] = []
    if since is not None:
        conds.append("occurred_at >= %s")
        params.append(since)
    if until is not None:
        conds.append("occurred_at < %s")
        params.append(until)
    if action:
        conds.append("action = %s")
        params.append(action)
    clause = (" WHERE " + " AND ".join(conds)) if conds else ""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT date_trunc('{trunc}', occurred_at) AS bkt, COUNT(*) FROM access_log{clause} "
            "GROUP BY bkt ORDER BY bkt ASC", params)
        buckets = [
            {"bucket": b.isoformat() if b is not None else None, "count": int(c)}
            for b, c in cur.fetchall()]
    return {"interval": trunc, "buckets": buckets}
