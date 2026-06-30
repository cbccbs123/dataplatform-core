"""013 US4 — 자산/FSM 대시보드 집계·목록 조회. 읽기 전용·결정적(헌법 3·6조)·LLM 0.

자산 데이터·스키마는 쓰기 0(SELECT only). 의료 도메인은 고정 SQL 로 항상 제외하며,
정렬은 COUNT(*) DESC + key ASC / created_at DESC + asset_id DESC tiebreak 으로 결정적이다.
"""
from __future__ import annotations

import os
from typing import Any

_EXCLUDE_MEDICAL = "domain_label <> 'medical'"  # 고정 SQL(사용자 입력 아님)·검색/상세와 일관
# 파일 확장자(file_type) = fs_path 마지막 .세그먼트(소문자·없으면 NULL). 고정 SQL·raw 정규식(인젝션 안전).
_EXT_EXPR = r"lower(substring(fs_path from '\.([^./]+)$'))"


def asset_stats(conn: Any) -> dict[str, Any]:
    """전체 자산 집계(status·modality·domain·file_type·date별·총계·의료 제외·결정적·FR-009e)."""
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
        by_file_type = [{"file_type": e, "count": int(c)} for e, c in cur.fetchall()]
        cur.execute(f"SELECT created_at::date AS d, COUNT(*) FROM asset WHERE {_EXCLUDE_MEDICAL} "
                    "GROUP BY d ORDER BY d ASC")
        by_date = [{"date": d.isoformat() if d is not None else None, "count": int(c)}
                   for d, c in cur.fetchall()]
    return {"total": total, "by_status": by_status, "by_modality": by_modality,
            "by_domain": by_domain, "by_file_type": by_file_type, "by_date": by_date}


def query_assets(conn: Any, *, status: str | None = None, modality: str | None = None,
                 domain: str | None = None, file_type: str | None = None,
                 created_from: Any = None, created_to: Any = None,
                 limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """자산 목록(FSM 단계·modality·domain·file_type·날짜 필터·페이징·의료 제외·created_at DESC·FR-009f)."""
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
    if file_type:
        conds.append(f"{_EXT_EXPR} = %s")
        params.append(file_type)
    if created_from is not None:
        conds.append("created_at >= %s")
        params.append(created_from)
    if created_to is not None:
        conds.append("created_at < %s")
        params.append(created_to)
    clause = " WHERE " + " AND ".join(conds)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM asset" + clause, params)
        total = int(cur.fetchone()[0])
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
