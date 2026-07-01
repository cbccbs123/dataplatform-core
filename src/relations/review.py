"""관계 HITL 검토 — proposed 엣지 승인/반려 + relation_kind 승격(inactive→active).

검토 큐 설계
    별도 큐 테이블 없이 ``graph_edge.status = 'proposed'`` 자체가 검토 큐다.
    ``list_proposed_edges`` 로 조회하고, ``approve_edge``/``reject_edge`` 로 상태를 전환한다.

멱등·반려 부활 방지 계약
    ``_decide_edge`` 의 ``WHERE edge_id = %s AND status = 'proposed'`` 가드가 핵심이다.
    이미 active 또는 rejected 인 엣지는 갱신 대상에서 제외되므로 rowcount == 0 이 되어
    False 를 반환한다. 소비자는 이 반환값으로 "이미 결정된 엣지" 여부를 판단한다.

관계 어휘 거버넌스
    ``promote_relation_kind`` 는 LLM 이 제안해 inactive 로 쌓인 kind 를 active 로 올리는
    HITL 경로다. 반대 방향(active→inactive 강등)은 현재 미구현 — 직접 SQL 로 처리한다.
"""
from __future__ import annotations

import os
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

# 조회·정정 status 화이트리스트(포탈 API 검증과 공유). 검토 큐 = proposed,
# 승인 내역 = active, 비승인 내역 = rejected 세 상태만 노출·전이 대상이다.
_REVIEW_STATUSES = ("proposed", "active", "rejected")

# 검토 목록 조회 SQL(FR-102) — node→asset 양끝 조인으로 식별 보강.
# C6: 검토 화면은 "원본 엣지 행" 단위라 graph_query seam 의 대칭 정규화(dst 접힘 복원)를
#     쓰지 않는다. 노출용 대칭 정규화는 그 seam 의 책임이고 검토(원본 행)와 무관하다.
# G7 확장: 고정 WHERE(_REVIEW_FROM_WHERE) 를 조인 상수(_REVIEW_FROM) + 동적 WHERE 빌더로
#   분리했다. 의료 제외 2개 + status 는 항상 붙고, 검색·필터·기간은 주어진 것만 append 한다.
#   COUNT 와 rows 가 동일 WHERE·params 를 공유하도록 빌더가 (conditions, params) 를 함께 만든다.
_REVIEW_FROM = """
FROM graph_edge e
JOIN relation_kind rk ON rk.relation_kind_id = e.relation_kind_id
JOIN node sn ON sn.node_id = e.src_node AND sn.node_kind = 'asset'
JOIN node dn ON dn.node_id = e.dst_node AND dn.node_kind = 'asset'
JOIN asset sa ON sa.asset_id = sn.asset_id
JOIN asset da ON da.asset_id = dn.asset_id
"""

# 기간 필터 date_col 화이트리스트 — f-string 으로 컬럼명을 조립하되, 이 상수 집합에 없는
# 값이면 진입 즉시 ValueError 로 거부해 인젝션을 원천 차단한다(013 _TIMELINE_INTERVALS 관례).
_REVIEW_DATE_COLS = ("created_at", "reviewed_at")


def _build_review_where(
    *,
    status: str,
    q: str | None,
    asset_id: str | None,
    kind_code: str | None,
    modality: str | None,
    min_confidence: float | None,
    max_confidence: float | None,
    reviewed_by: str | None,
    since: Any,
    until: Any,
    date_col: str,
) -> tuple[str, list[Any]]:
    """동적 WHERE 절(문자열) + params 리스트를 만든다(FR-701~753).

    의료(PHI) 제외 2개 + ``e.status = %s`` 는 항상 붙는다(헌법 10조·NULL 도메인 노출 위해
    ``IS DISTINCT FROM`` 사용 — ``= 'medical'`` 은 NULL 을 놓친다). 그 밖의 검색·필터·기간
    조건은 인자가 주어졌을 때만 append 한다 → 미지정 시 현행과 완전 동일(하위 호환·SC-011).
    모든 값은 %s 바인딩(인젝션 0). ``date_col`` 만 f-string 조립이라 화이트리스트로 검증한다.
    """
    if date_col not in _REVIEW_DATE_COLS:
        raise ValueError(f"date_col 은 {_REVIEW_DATE_COLS} 만 허용: {date_col!r}")

    conditions: list[str] = [
        "e.status = %s",
        "sa.domain_label IS DISTINCT FROM 'medical'",
        "da.domain_label IS DISTINCT FROM 'medical'",
    ]
    params: list[Any] = [status]

    if q:
        # 통합 텍스트 검색(FR-702) — 대소문자 무시 부분 일치를 8개 필드에 OR 매칭한다.
        # UUID 컬럼은 ::text 캐스팅 후 ILIKE. Postgres 에 basename() 이 없어 fs_path 전체를
        # ILIKE 하는데, 파일명이 경로의 부분문자열이라 파일명 검색을 커버한다.
        q_pat = f"%{q}%"
        conditions.append(
            "(e.edge_id::text ILIKE %s OR sn.asset_id::text ILIKE %s"
            " OR dn.asset_id::text ILIKE %s OR sa.fs_path ILIKE %s"
            " OR da.fs_path ILIKE %s OR e.reason ILIKE %s"
            " OR e.topic->>'topic_ko' ILIKE %s OR e.topic->>'topic_en' ILIKE %s)"
        )
        params.extend([q_pat] * 8)
    if asset_id is not None:
        # 양끝 중 하나 정확 일치. ::text = 로 비교해 비-UUID 입력에도 500 아닌 0건이 되게 한다.
        conditions.append("(sn.asset_id::text = %s OR dn.asset_id::text = %s)")
        params.extend([asset_id, asset_id])
    if kind_code is not None:
        # 미지 값도 검증 없이 그대로 바인딩 → 200 + total=0(FR-703).
        conditions.append("rk.kind_code = %s")
        params.append(kind_code)
    if modality is not None:
        conditions.append("(sa.modality = %s OR da.modality = %s)")
        params.extend([modality, modality])
    if min_confidence is not None:
        conditions.append("e.confidence >= %s")
        params.append(min_confidence)
    if max_confidence is not None:
        conditions.append("e.confidence <= %s")
        params.append(max_confidence)
    if reviewed_by is not None:
        conditions.append("e.reviewed_by::text = %s")
        params.append(reviewed_by)
    if since is not None:
        # date_col 은 위에서 화이트리스트 검증됨 → f-string 안전. 값(since)은 %s 바인딩.
        conditions.append(f"e.{date_col} >= %s")
        params.append(since)
    if until is not None:
        conditions.append(f"e.{date_col} < %s")  # exclusive(FR-751)
        params.append(until)

    return "WHERE " + "\n  AND ".join(conditions), params


def list_edges_for_review(
    conn: Connection[Any],
    *,
    status: str = "proposed",
    limit: int = 50,
    offset: int = 0,
    q: str | None = None,
    asset_id: str | None = None,
    kind_code: str | None = None,
    modality: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    reviewed_by: str | None = None,
    since: Any = None,
    until: Any = None,
    date_col: str = "created_at",
) -> dict[str, Any]:
    """status별 엣지를 식별 보강해 페이징 조회한다(FR-101/102/103 + G7 검색·필터·기간).

    각 행에 양끝 자산(asset_id·file_name·modality)·kind_code·confidence·reason·topic·
    status·reviewed_by·reviewed_at·created_at 를 담아 UI 가 "무엇을 승인하는지" 알 수 있게
    한다(CR-11/CR-18 최소 해소). ``file_name`` 은 asset 에 컬럼이 없어 ``fs_path`` basename
    으로 파생한다(``src/portal/download.py`` 관례 일치).

    선택 인자(``q``·``asset_id``·``kind_code``·``modality``·``min/max_confidence``·
    ``reviewed_by``·``since``/``until``+``date_col``)는 주어진 것만 WHERE 에 AND 조합한다.
    **전부 생략하면 현행과 완전 동일**(하위 호환·SC-011). status/필터 값 검증은 호출자(포탈
    API) 책임이고, ``date_col`` 만 함수가 화이트리스트 검증한다(f-string 조립·인젝션 방지).
    total 은 같은 WHERE·params 로 COUNT 해 페이징 UI(proposed 4.7k 큐·필터 후 total 일치·
    FR-705)를 받친다. 반환 ``{rows, total, status, limit, offset}``.
    """
    where, params = _build_review_where(
        status=status, q=q, asset_id=asset_id, kind_code=kind_code, modality=modality,
        min_confidence=min_confidence, max_confidence=max_confidence,
        reviewed_by=reviewed_by, since=since, until=until, date_col=date_col,
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT COUNT(*) AS count\n" + _REVIEW_FROM + where, tuple(params))
        total = int(cur.fetchone()["count"])
        cur.execute(
            """
            SELECT e.edge_id, rk.kind_code, e.confidence, e.reason, e.topic, e.status,
                   e.reviewed_by, e.reviewed_at, e.created_at,
                   sn.asset_id AS src_asset_id, sa.fs_path AS src_fs_path,
                   sa.modality AS src_modality,
                   dn.asset_id AS dst_asset_id, da.fs_path AS dst_fs_path,
                   da.modality AS dst_modality
            """
            + _REVIEW_FROM
            + where
            + """
            ORDER BY e.confidence DESC NULLS LAST, e.edge_id
            LIMIT %s OFFSET %s
            """,
            tuple(params) + (limit, offset),
        )
        rows = [_review_row(r) for r in cur.fetchall()]
    return {"rows": rows, "total": total, "status": status, "limit": limit, "offset": offset}


def _review_row(r: dict[str, Any]) -> dict[str, Any]:
    """조회 행 → 검토 목록 shape(src/dst 각 {asset_id, file_name, modality}).

    edge_id·asset_id 는 ``str`` 로 정규화한다 — psycopg 는 UUID 컬럼을 ``uuid.UUID`` 객체로
    반환하므로, ``graph_query`` seam 관례(edge_id/asset_id str화)와 맞춰 파이썬 소비자의 문자열
    비교·JSON 직렬화 일관성을 보장한다(미변환 시 ``UUID(...) == "..."`` 가 False 가 되는 함정).
    ``reviewed_by`` 는 NULL(미결정 엣지) 가능이라 str화하지 않는다(None → 'None' 방지).
    ``created_at`` 은 datetime 그대로 둔다(NOT NULL·FastAPI 가 ISO 8601 로 직렬화·FR-761).
    """
    return {
        "edge_id": str(r["edge_id"]),
        "kind_code": r["kind_code"],
        "confidence": r["confidence"],
        "reason": r["reason"],
        "topic": r["topic"],
        "status": r["status"],
        "reviewed_by": r["reviewed_by"],
        "reviewed_at": r["reviewed_at"],
        "created_at": r["created_at"],
        "src": {
            "asset_id": str(r["src_asset_id"]),
            "file_name": os.path.basename(r["src_fs_path"] or ""),
            "modality": r["src_modality"],
        },
        "dst": {
            "asset_id": str(r["dst_asset_id"]),
            "file_name": os.path.basename(r["dst_fs_path"] or ""),
            "modality": r["dst_modality"],
        },
    }


def list_relation_kinds(
    conn: Connection[Any], *, status: str | None = None
) -> dict[str, Any]:
    """relation_kind 목록을 조회한다(FR-801·필터 드롭다운용·조회 전용·LLM 0).

    ``{rows:[{kind_code, kind_name_ko, status}], total}`` 를 반환한다. ``ORDER BY kind_code``
    로 결정적 정렬(헌법 3조). ``status`` 지정 시 ``WHERE status = %s``(active|inactive 화이트
    리스트 검증은 호출자 책임), 미지정이면 전체. relation_kind 테이블 재사용(마이그레이션 0).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        if status is not None:
            cur.execute(
                "SELECT kind_code, kind_name_ko, status FROM relation_kind"
                " WHERE status = %s ORDER BY kind_code",
                (status,),
            )
        else:
            cur.execute(
                "SELECT kind_code, kind_name_ko, status FROM relation_kind"
                " ORDER BY kind_code"
            )
        rows = [dict(r) for r in cur.fetchall()]
    return {"rows": rows, "total": len(rows)}


def list_proposed_edges(conn: Connection[Any], *, limit: int = 100) -> list[dict[str, Any]]:
    """검토 대기(proposed) 엣지를 신뢰도 높은 순으로. status='active' 필터는 소비자 몫.

    ``topic`` 컬럼은 jsonb(topic_ko/en, subtopic_ko/en)다. 소비자가 필요한 키를 꺼내 쓴다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT e.edge_id, e.src_node, e.dst_node, rk.kind_code,
                   e.confidence, e.reason, e.topic
            FROM graph_edge e
            JOIN relation_kind rk ON rk.relation_kind_id = e.relation_kind_id
            WHERE e.status = 'proposed'
            ORDER BY e.confidence DESC NULLS LAST
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def _decide_edge(conn: Connection[Any], *, edge_id: str, reviewer: str, status: str) -> bool:
    """proposed 엣지만 status 로 확정(이미 결정된 엣지는 변경 안 함). 1행 갱신 시 True.

    ``AND status = 'proposed'`` 가드 목적
        - 이미 active 인 엣지를 실수로 재반려하는 사고를 차단한다.
        - 이미 rejected 인 엣지가 재승인 요청으로 부활하는 것을 막는다.
        - rowcount == 1 확인으로 "내가 이 결정을 실제로 수행했는가"를 반환한다.
          rowcount == 0 은 "엣지 없음" 또는 "이미 결정됨" 두 경우를 포함한다.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE graph_edge
            SET status = %s, reviewed_by = %s, reviewed_at = now(), updated_at = now()
            WHERE edge_id = %s AND status = 'proposed'
            """,
            (status, reviewer, edge_id),
        )
        return cur.rowcount == 1


def approve_edge(conn: Connection[Any], *, edge_id: str, reviewer: str) -> bool:
    """proposed 엣지를 active 로 승인 — 이후 graph_query 의 status='active' 필터에 잡혀 그래프에 노출된다.

    1행 갱신 시 True(계약·멱등·proposed 가드는 ``_decide_edge`` 참조).
    """
    return _decide_edge(conn, edge_id=edge_id, reviewer=reviewer, status="active")


def reject_edge(conn: Connection[Any], *, edge_id: str, reviewer: str) -> bool:
    """proposed 엣지를 rejected 로 반려 — **소프트**(삭제 아님, status 전이만)라 status 필터에서 빠질 뿐 행은 남는다.

    이 rejected 결정은 사람의 판단이므로, 이후 LLM 이 같은 쌍을 재제안해도
    ``sync_graph_edges`` 의 ON CONFLICT 가 status 를 덮지 않아 rejected 가 보존된다(부활 방지).
    1행 갱신 시 True(계약은 ``_decide_edge`` 참조).
    """
    return _decide_edge(conn, edge_id=edge_id, reviewer=reviewer, status="rejected")


def bulk_review(
    conn: Connection[Any], *, edge_ids: list[str], reviewer: str, action: str
) -> list[dict[str, Any]]:
    """일괄 승인/반려 — 건별 기존 단건 함수를 호출하고 per-id 결과를 모은다(FR-201/202/203).

    ``action`` 은 "approve"|"reject" — 각각 검증된 ``approve_edge``/``reject_edge`` 로
    디스패치한다(재구현 0·proposed 가드·멱등은 단건 함수 계약 그대로). 반환은
    ``[{"edge_id", "ok"}]`` per-id 배열이다.

    한 트랜잭션 = 호출자(포탈 ``_run_in_db_write``)가 커밋한다 — 여기서는 커밋/롤백하지
    않는다. ``ok=False``(엣지 없음 또는 이미 결정됨=proposed 아님)는 **예외가 아니라 결과값**
    이므로 나머지 엣지 처리를 멈추지 않는다. 성공(ok=True) 건은 같은 트랜잭션에서 원자적으로
    함께 커밋된다(FR-203).

    ``action`` 화이트리스트 가드: 오타·미지 값이 조용히 reject 로 처리되는 사고를 막는다
    (놀람 최소화 — approve/reject 외에는 즉시 ValueError).
    """
    if action not in ("approve", "reject"):
        raise ValueError(f"action 은 'approve'|'reject' 만 허용: {action!r}")
    decide = approve_edge if action == "approve" else reject_edge
    return [
        {"edge_id": eid, "ok": decide(conn, edge_id=eid, reviewer=reviewer)}
        for eid in edge_ids
    ]


def revise_edge(
    conn: Connection[Any], *, edge_id: str, reviewer: str, to_status: str
) -> bool:
    """사람 전용 결정 정정 — proposed 가드 **없이** status 를 전이한다(FR-301·C4).

    ``_decide_edge`` 의 ``AND status = 'proposed'`` 가드를 **우회하는 유일 경로**다. 운영에서
    오결정(잘못 approve/reject)을 되돌리려면 active↔rejected·→proposed 전 방향 전이가 필요한데,
    proposed 가드가 있으면 이미 결정된 엣지를 되돌릴 수 없다. 그래서 이 함수만 가드 없이
    ``WHERE edge_id = %s`` 로 갱신하고 ``reviewed_by``/``reviewed_at``/``updated_at`` 을 새로 찍는다.

    LLM 경로(``sync_graph_edges``)와 분리(FR-302): revise 가 status 를 바꿔도, LLM 재제안의
    ON CONFLICT ``DO UPDATE SET`` 는 여전히 ``status`` 를 **미갱신**한다(graph_persist.py). 즉
    사람의 정정이 LLM 재제안에 덮이지 않는다 — 사람↔LLM 경계는 그대로 보존된다.

    to_status 화이트리스트 검증은 호출자(포탈 API·``_REVIEW_STATUSES``) 책임이다.
    1행 갱신 시 True, 대상 엣지 없음(rowcount==0) 시 False.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE graph_edge
            SET status = %s, reviewed_by = %s, reviewed_at = now(), updated_at = now()
            WHERE edge_id = %s
            """,
            (to_status, reviewer, edge_id),
        )
        return cur.rowcount == 1


def promote_relation_kind(conn: Connection[Any], *, kind_code: str, reviewer: str) -> bool:
    """LLM 제안으로 쌓인 inactive relation_kind 를 active 로 승격(어휘 거버넌스). reviewer 는 lineage 기록용.

    ``AND status = 'inactive'`` 가드
        이미 active 인 kind 를 중복 승격해도 rowcount == 0 이 되어 False 를 반환한다(멱등).

    reviewer 미저장 이유
        현재 relation_kind 스키마에 reviewed_by 컬럼이 없다.
        ``_ = reviewer`` 는 시그니처는 유지하면서 미래 lineage 컬럼 추가를 위한 자리 표시자다.
    """
    _ = reviewer  # 향후 relation_kind.reviewed_by 컬럼 추가 시 여기에 저장
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE relation_kind SET status='active' WHERE kind_code=%s AND status='inactive'",
            (kind_code,),
        )
        return cur.rowcount == 1
