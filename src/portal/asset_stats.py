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
_GROUP_COLS = {"modality": "modality", "status": "status", "domain": "domain_label"}


def asset_stats(conn: Any) -> dict[str, Any]:
    """전체 자산 집계(status·modality·domain·file_ext·date별·총계·의료 제외·결정적·FR-009e)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM asset WHERE {_EXCLUDE_MEDICAL}")
        total = int(cur.fetchone()[0])
        cur.execute(f"SELECT status, COUNT(*) FROM asset WHERE {_EXCLUDE_MEDICAL} "
                    "GROUP BY status ORDER BY COUNT(*) DESC, status ASC")
        by_status = [{"status": s, "count": int(c)} for s, c in cur.fetchall()]
        cur.execute(f"SELECT modality, COUNT(*) FROM asset WHERE {_EXCLUDE_MEDICAL} "
                    "GROUP BY modality ORDER BY COUNT(*) DESC, modality ASC")
        by_modality = [{"modality": m, "count": int(c)} for m, c in cur.fetchall()]
        cur.execute(f"SELECT domain_label, COUNT(*) FROM asset WHERE {_EXCLUDE_MEDICAL} "
                    "GROUP BY domain_label ORDER BY COUNT(*) DESC, domain_label ASC")
        by_domain = [{"domain": d, "count": int(c)} for d, c in cur.fetchall()]
        cur.execute(f"SELECT {_EXT_EXPR} AS ext, COUNT(*) FROM asset WHERE {_EXCLUDE_MEDICAL} "
                    "GROUP BY ext ORDER BY COUNT(*) DESC, ext ASC NULLS LAST")
        by_file_ext = [{"file_ext": e, "count": int(c)} for e, c in cur.fetchall()]
        cur.execute(f"SELECT created_at::date AS d, COUNT(*) FROM asset WHERE {_EXCLUDE_MEDICAL} "
                    "GROUP BY d ORDER BY d ASC")
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
    동반(모달리티 상세에서 자산을 안 열고도 내용 파악). 기본은 가벼운 목록(하위호환). WHERE 절 컬럼은
    asset 에만 있어 한정 불요하나, JOIN 시 asset_id 모호 방지로 SELECT/ORDER BY 는 ``a.`` 한정.
    """
    conds = [_EXCLUDE_MEDICAL]
    params: list[Any] = []
    if status:
        conds.append("status = %s")
        params.append(status)
    if modality:
        conds.append("modality = %s")
        params.append(modality)
    if domain:
        conds.append("domain_label = %s")
        params.append(domain)
    if file_ext:
        conds.append(f"{_EXT_EXPR} = %s")
        params.append(file_ext)
    if created_from is not None:
        conds.append("created_at >= %s")
        params.append(created_from)
    if created_to is not None:
        conds.append("created_at < %s")
        params.append(created_to)
    clause = " WHERE " + " AND ".join(conds)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM asset" + clause, params)  # COUNT 은 JOIN 불요(경량)
        total = int(cur.fetchone()[0])
        if with_content:
            cur.execute(
                "SELECT a.asset_id, a.status, a.modality, a.domain_label, a.fs_path, a.created_at, "
                "m.ext_meta->>'summary' AS summary, m.ext_meta->'keywords' AS keywords "
                "FROM asset a LEFT JOIN asset_metadata m ON m.asset_id = a.asset_id"
                + clause + " ORDER BY a.created_at DESC, a.asset_id DESC LIMIT %s OFFSET %s",
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
                + clause + " ORDER BY created_at DESC, asset_id DESC LIMIT %s OFFSET %s",
                [*params, limit, offset])
            rows = [
                {"asset_id": str(aid), "status": st, "modality": mod, "domain_label": dl,
                 "file_name": os.path.basename(fp) if fp else None,
                 "created_at": ts.isoformat() if ts is not None else None}
                for aid, st, mod, dl, fp, ts in cur.fetchall()]
    return {"rows": rows, "total": total}


def modality_detail(conn: Any, modality: str) -> dict[str, Any]:
    """단일 모달리티 스코프 집계(보완 v6) — 총계 + 확장자·상태·일자별. 의료 제외·결정적·LLM 0.

    모달리티 드릴다운(예: video 안에서 mp4/mov 분포·일자 추이·FSM 상태). modality 는 %s 바인딩.
    """
    where = f"WHERE {_EXCLUDE_MEDICAL} AND modality = %s"
    p = [modality]
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
