"""013 US4 — 자산/FSM 대시보드 집계·목록 조회. 읽기 전용·결정적(헌법 3·6조)·LLM 0.

자산 데이터·스키마는 쓰기 0(SELECT only). 의료 도메인은 고정 SQL 로 항상 제외하며,
정렬은 COUNT(*) DESC + key ASC / created_at DESC + asset_id DESC tiebreak 으로 결정적이다.
"""
from __future__ import annotations

import os
from typing import Any

from src.portal._timeline_util import pivot_series

_EXCLUDE_MEDICAL = "domain_label <> 'medical'"  # 고정 SQL(사용자 입력 아님)·검색/상세와 일관
# 파일 확장자(file_ext) = fs_path 마지막 .세그먼트(소문자·없으면 NULL). 고정 SQL·raw 정규식(인젝션 안전).
_EXT_EXPR = r"lower(substring(fs_path from '\.([^./]+)$'))"
_INTERVALS = {"day", "hour"}  # date_trunc 화이트리스트(f-string 인젝션 방지)
# 자산 생성 추이 group_by 화이트리스트 → 컬럼식(고정 매핑·사용자 입력은 키로만 조회·인젝션 안전).
# file_ext 는 평컬럼이 아닌 확장자 정규식(_EXT_EXPR) — 단일 테이블(asset) 쿼리라 fs_path 비한정 안전.
_GROUP_COLS = {"modality": "modality", "status": "status", "domain": "domain_label",
               "file_ext": _EXT_EXPR}


def _period_clause(since: Any, until: Any) -> tuple[str, list[Any]]:
    """의료 제외 + 생성일(created_at) 기간 필터 WHERE 절·파라미터(단일 테이블 asset 전용·비한정).

    to(until) 는 exclusive(``< %s``) — query_assets·timeline·다른 API 와 동일 규칙. 미지정이면 전체.
    """
    conds = [_EXCLUDE_MEDICAL]
    params: list[Any] = []
    if since is not None:
        conds.append("created_at >= %s")
        params.append(since)
    if until is not None:
        conds.append("created_at < %s")
        params.append(until)
    return "WHERE " + " AND ".join(conds), params


def asset_stats(conn: Any, *, since: Any = None, until: Any = None) -> dict[str, Any]:
    """전체 자산 집계(status·modality·domain·file_ext·date별·총계·의료 제외·결정적·FR-009e).

    ``since``/``until``(생성일 from/to·to exclusive·보완 v6) 지정 시 6개 집계 전부 기간 스코프
    (대시보드 기간 필터가 파일 포맷·모달리티·일자 분포에 일관 반영). 미지정이면 전체 기간.
    """
    where, p = _period_clause(since, until)
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM asset {where}", p)
        total = int(cur.fetchone()[0])
        cur.execute(f"SELECT status, COUNT(*) FROM asset {where} "
                    "GROUP BY status ORDER BY COUNT(*) DESC, status ASC", p)
        by_status = [{"status": s, "count": int(c)} for s, c in cur.fetchall()]
        cur.execute(f"SELECT modality, COUNT(*) FROM asset {where} "
                    "GROUP BY modality ORDER BY COUNT(*) DESC, modality ASC", p)
        by_modality = [{"modality": m, "count": int(c)} for m, c in cur.fetchall()]
        cur.execute(f"SELECT domain_label, COUNT(*) FROM asset {where} "
                    "GROUP BY domain_label ORDER BY COUNT(*) DESC, domain_label ASC", p)
        by_domain = [{"domain": d, "count": int(c)} for d, c in cur.fetchall()]
        cur.execute(f"SELECT {_EXT_EXPR} AS ext, COUNT(*) FROM asset {where} "
                    "GROUP BY ext ORDER BY COUNT(*) DESC, ext ASC NULLS LAST", p)
        by_file_ext = [{"file_ext": e, "count": int(c)} for e, c in cur.fetchall()]
        cur.execute(f"SELECT created_at::date AS d, COUNT(*) FROM asset {where} "
                    "GROUP BY d ORDER BY d ASC", p)
        by_date = [{"date": d.isoformat() if d is not None else None, "count": int(c)}
                   for d, c in cur.fetchall()]
    return {"total": total, "by_status": by_status, "by_modality": by_modality,
            "by_domain": by_domain, "by_file_ext": by_file_ext, "by_date": by_date}


def query_assets(conn: Any, *, status: str | None = None, modality: str | None = None,
                 domain: str | None = None, file_ext: str | None = None,
                 created_from: Any = None, created_to: Any = None,
                 limit: int = 50, offset: int = 0, with_content: bool = False) -> dict[str, Any]:
    """자산 목록(FSM 단계·modality·domain·file_ext·날짜 필터·페이징·의료 제외·created_at DESC·FR-009f).

    ``with_content=True``(보완 v6) — asset_metadata LEFT JOIN 으로 행마다 요약·키워드(+제목=파일명)
    동반(모달리티 상세에서 자산을 안 열고도 내용 파악). 메타 미적재 자산은 LEFT JOIN 으로 행은 남되
    summary/keywords 가 None. 기본은 가벼운 목록(하위호환).

    **모호성 주의**: ``asset`` 과 ``asset_metadata`` 둘 다 ``asset_id``·``created_at`` 컬럼을 가져,
    content 경로의 JOIN 에서 비한정 컬럼은 PG 오류가 난다. 그래서 content WHERE/SELECT/ORDER BY 는
    ``a.`` 한정 절을 따로 쓴다(COUNT·비콘텐츠 SELECT 는 단일 테이블이라 비한정 유지).
    """
    def _conds(pfx: str) -> str:
        # 접두사(pfx)만 다른 동일 조건 — 파라미터는 pfx 무관(아래 params 와 동일 순서).
        ext = rf"lower(substring({pfx}fs_path from '\.([^./]+)$'))"
        c = [f"{pfx}{_EXCLUDE_MEDICAL}"]
        if status:
            c.append(f"{pfx}status = %s")
        if modality:
            c.append(f"{pfx}modality = %s")
        if domain:
            c.append(f"{pfx}domain_label = %s")
        if file_ext:
            c.append(f"{ext} = %s")
        if created_from is not None:
            c.append(f"{pfx}created_at >= %s")
        if created_to is not None:
            c.append(f"{pfx}created_at < %s")
        return " WHERE " + " AND ".join(c)

    params: list[Any] = []  # _conds 의 조건 추가 순서와 1:1 (medical 은 파라미터 없음)
    if status:
        params.append(status)
    if modality:
        params.append(modality)
    if domain:
        params.append(domain)
    if file_ext:
        params.append(file_ext)
    if created_from is not None:
        params.append(created_from)
    if created_to is not None:
        params.append(created_to)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM asset" + _conds(""), params)  # COUNT 은 JOIN 불요(경량·비한정)
        total = int(cur.fetchone()[0])
        if with_content:
            cur.execute(
                "SELECT a.asset_id, a.status, a.modality, a.domain_label, a.fs_path, a.created_at, "
                "m.ext_meta->>'summary' AS summary, m.ext_meta->'keywords' AS keywords "
                "FROM asset a LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id"
                + _conds("a.") + " ORDER BY a.created_at DESC, a.asset_id DESC LIMIT %s OFFSET %s",
                [*params, limit, offset])
            rows = [
                {"asset_id": str(aid), "status": st, "modality": mod, "domain_label": dl,
                 "file_name": os.path.basename(fp) if fp else None,
                 "created_at": ts.isoformat() if ts is not None else None,
                 "summary": summary, "keywords": kw}
                for aid, st, mod, dl, fp, ts, summary, kw in cur.fetchall()]
        else:
            cur.execute(
                "SELECT asset_id, status, modality, domain_label, fs_path, created_at FROM asset"
                + _conds("") + " ORDER BY created_at DESC, asset_id DESC LIMIT %s OFFSET %s",
                [*params, limit, offset])
            rows = [
                {"asset_id": str(aid), "status": st, "modality": mod, "domain_label": dl,
                 "file_name": os.path.basename(fp) if fp else None,
                 "created_at": ts.isoformat() if ts is not None else None}
                for aid, st, mod, dl, fp, ts in cur.fetchall()]
    return {"rows": rows, "total": total}


def modality_detail(conn: Any, modality: str, *, since: Any = None,
                    until: Any = None) -> dict[str, Any]:
    """단일 모달리티 스코프 집계(보완 v6) — 총계 + 확장자·상태·일자별. 의료 제외·결정적·LLM 0.

    모달리티 드릴다운(예: video 안에서 mp4/mov 분포·일자 추이·FSM 상태). modality 는 %s 바인딩.
    ``since``/``until``(생성일 from/to·to exclusive) 지정 시 개요(asset_stats) 기간 필터와 일관 스코프.
    """
    conds = [_EXCLUDE_MEDICAL, "modality = %s"]
    p: list[Any] = [modality]
    if since is not None:
        conds.append("created_at >= %s")
        p.append(since)
    if until is not None:
        conds.append("created_at < %s")
        p.append(until)
    where = "WHERE " + " AND ".join(conds)
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM asset {where}", p)
        total = int(cur.fetchone()[0])
        cur.execute(f"SELECT {_EXT_EXPR} AS ext, COUNT(*) FROM asset {where} "
                    "GROUP BY ext ORDER BY COUNT(*) DESC, ext ASC NULLS LAST", p)
        by_file_ext = [{"file_ext": e, "count": int(c)} for e, c in cur.fetchall()]
        cur.execute(f"SELECT status, COUNT(*) FROM asset {where} "
                    "GROUP BY status ORDER BY COUNT(*) DESC, status ASC", p)
        by_status = [{"status": s, "count": int(c)} for s, c in cur.fetchall()]
        cur.execute(f"SELECT created_at::date AS d, COUNT(*) FROM asset {where} "
                    "GROUP BY d ORDER BY d ASC", p)
        by_date = [{"date": d.isoformat() if d is not None else None, "count": int(c)}
                   for d, c in cur.fetchall()]
    return {"modality": modality, "total": total, "by_file_ext": by_file_ext,
            "by_status": by_status, "by_date": by_date}


def asset_timeline(conn: Any, *, since: Any = None, until: Any = None,
                   interval: str = "day", group_by: str | None = None) -> dict[str, Any]:
    """자산 생성 일자 추이(보완 v6·계보 timeline 과 동일 멀티시리즈 패턴). 의료 제외·결정적·LLM 0.

    ``group_by``(modality/status/domain) 주면 멀티시리즈(시리즈 key ASC·버킷 ASC), 미지정이면 단일
    시리즈({interval, buckets}). trunc 화이트리스트(f-string 안전)·기간(since/until)은 %s 바인딩.
    """
    trunc = interval if interval in _INTERVALS else "day"
    conds = [_EXCLUDE_MEDICAL]
    params: list[Any] = []
    if since is not None:
        conds.append("created_at >= %s")
        params.append(since)
    if until is not None:
        conds.append("created_at < %s")
        params.append(until)
    where = " WHERE " + " AND ".join(conds)
    with conn.cursor() as cur:
        if group_by in _GROUP_COLS:
            gcol = _GROUP_COLS[group_by]
            cur.execute(
                f"SELECT {gcol} AS key, date_trunc('{trunc}', created_at) AS bkt, COUNT(*) "
                f"FROM asset{where} GROUP BY key, bkt ORDER BY key ASC, bkt ASC", params)
            return {"interval": trunc, "group_by": group_by, "series": pivot_series(cur.fetchall())}
        cur.execute(f"SELECT date_trunc('{trunc}', created_at) AS bkt, COUNT(*) "
                    f"FROM asset{where} GROUP BY bkt ORDER BY bkt ASC", params)
        buckets = [{"bucket": b.isoformat() if b is not None else None, "count": int(c)}
                   for b, c in cur.fetchall()]
        return {"interval": trunc, "buckets": buckets}
