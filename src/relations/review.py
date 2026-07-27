"""사람이 관계를 검토하는 계층 — 제안된 엣지의 승인·반려, 새 관계 종류의 승격.

검토 큐 설계
    별도 큐 테이블이 없다 — ``status = 'proposed'`` 인 엣지 자체가 검토 대기 목록이다.
    조회는 ``list_edges_for_review``, 결정은 ``approve_edge``/``reject_edge`` 가 맡는다.

멱등·반려 부활 방지 계약
    ``_decide_edge`` 의 ``WHERE edge_id = %s AND status = 'proposed'`` 가드가 핵심이다.
    이미 active 또는 rejected 인 엣지는 갱신 대상에서 제외되므로 rowcount == 0 이 되어
    False 를 반환한다. 소비자는 이 반환값으로 "이미 결정된 엣지" 여부를 판단한다.

관계 어휘 거버넌스
    ``promote_relation_kind`` 는 LLM 이 제안해 쌓인 관계 종류를 사람이 승인해 쓰이게 하는
    유일한 경로다. 반대(승인 취소)는 아직 함수가 없어 직접 SQL 로 처리해야 한다.
"""
from __future__ import annotations

import os
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.relations.path_signal import like_escape  # LIKE 메타문자 이스케이프 공용(B9·SSOT)

# 노출·전이가 허용되는 상태 셋. 검토 대기(proposed) · 승인(active) · 반려(rejected).
_REVIEW_STATUSES = ("proposed", "active", "rejected")

# 검토 목록 조회의 고정 조인 부분. 양 끝 자산을 함께 끌어와 화면에 파일명·모달리티를 보여준다.
# ⚠️ 검토 화면은 **저장된 원본 행**을 그대로 보여준다 — 대칭 엣지를 질의 자산 관점으로 뒤집는
#    정규화는 조회 seam 의 몫이고, 여기서 하면 사람이 승인하려는 행과 화면이 어긋난다.
# WHERE 는 빌더가 따로 만든다. 목록과 COUNT 가 **같은 조건·같은 파라미터**를 써야 총계가 맞는다.
_REVIEW_FROM = """
FROM graph_edge e
JOIN relation_kind rk ON rk.relation_kind_id = e.relation_kind_id
JOIN node sn ON sn.node_id = e.src_node AND sn.node_kind = 'asset'
JOIN node dn ON dn.node_id = e.dst_node AND dn.node_kind = 'asset'
JOIN asset sa ON sa.asset_id = sn.asset_id
JOIN asset da ON da.asset_id = dn.asset_id
"""

# 기간 필터가 쓸 수 있는 컬럼. 컬럼명은 파라미터로 바인딩할 수 없어 문자열로 조립하므로,
# 이 목록 밖 값이 들어오면 즉시 거부해야 한다(그러지 않으면 인젝션 통로가 된다).
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
    """주어진 필터만 골라 WHERE 절과 파라미터를 만든다.

    상태 조건만 항상 붙고, 나머지는 값이 주어졌을 때만 추가된다.
    모든 값은 %s 바인딩(인젝션 0). ``date_col`` 만 f-string 조립이라 화이트리스트로 검증한다.

    Args:
        status: 엣지 상태(항상 조건에 붙는 유일한 필수 값).
        q: 통합 검색어. 주면 8개 필드에 대소문자 무시 부분 일치(OR)로 매칭한다.
            ``%``·``_`` 는 리터럴로 이스케이프된다.
        asset_id: 양끝 자산 중 하나와 정확 일치. 비-UUID 문자열이어도 오류 없이 0건이 된다.
        kind_code: 관계 종류 코드 정확 일치. 존재하지 않는 값도 검증 없이 0건으로 흘린다.
        modality: 양끝 자산 중 하나의 모달리티 일치.
        min_confidence: 신뢰도 하한(이상). ``None`` 이면 하한 없음.
        max_confidence: 신뢰도 상한(이하). ``None`` 이면 상한 없음.
        reviewed_by: 검토자 정확 일치.
        since: 기간 시작(**포함**). ``date_col`` 기준.
        until: 기간 끝(**미포함** — 하루 단위 조회에서 경계 중복을 피하려는 의도).
        date_col: 기간 필터가 볼 컬럼. ``created_at``(생성 시각) 또는 ``reviewed_at``(검토 시각).

    Returns:
        ``("WHERE ...", params)``. COUNT 와 목록 조회가 **같은 값을 공유**해야 total 이 맞는다.

    Raises:
        ValueError: ``date_col`` 이 화이트리스트 밖일 때(컬럼명은 바인딩할 수 없어 f-string 으로
            조립하므로, 여기서 막지 않으면 인젝션 통로가 된다).
    """
    if date_col not in _REVIEW_DATE_COLS:
        raise ValueError(f"date_col 은 {_REVIEW_DATE_COLS} 만 허용: {date_col!r}")

    conditions: list[str] = [
        "e.status = %s",
    ]
    params: list[Any] = [status]

    if q:
        # 한 검색어로 8개 필드를 한꺼번에 훑는다(대소문자 무시 부분 일치).
        # 경로 전체를 매칭하는 이유: DB 에 basename 함수가 없는데 파일명은 경로의 일부라,
        # 경로를 훑으면 파일명 검색도 함께 커버된다.
        # 검색어의 ``%``·``_`` 는 이스케이프한다 — 안 하면 "100%" 검색이 와일드카드로 동작한다.
        q_pat = f"%{like_escape(q)}%"
        conditions.append(
            "(e.edge_id::text ILIKE %s OR sn.asset_id::text ILIKE %s"
            " OR dn.asset_id::text ILIKE %s OR sa.fs_path ILIKE %s"
            " OR da.fs_path ILIKE %s OR e.reason ILIKE %s"
            " OR e.topic->>'topic_ko' ILIKE %s OR e.topic->>'topic_en' ILIKE %s)"
        )
        params.extend([q_pat] * 8)
    if asset_id is not None:
        # 양 끝 중 하나와 정확 일치. 문자열로 비교하므로 UUID 가 아닌 값이 와도 오류 대신 0건이다.
        conditions.append("(sn.asset_id::text = %s OR dn.asset_id::text = %s)")
        params.extend([asset_id, asset_id])
    if kind_code is not None:
        # 없는 코드가 와도 검증하지 않는다 — 오류 대신 0건으로 응답한다.
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
        conditions.append(f"e.{date_col} < %s")  # 끝 경계는 **미포함**(하루 단위 조회의 중복 방지)
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
    """상태별 엣지를 페이징 조회한다 — 양 끝 자산 정보까지 담아 화면이 바로 그릴 수 있게.

    검토자가 "무엇을 승인하는지" 알려면 엣지 정보만으로는 부족해 양 끝 자산의 파일명·모달리티까지
    싣는다. 파일명 컬럼은 따로 없어 경로에서 파생한다.

    조회 전용(쓰기 없음). 선택 인자는 **주어진 것만** WHERE 에 AND 로 붙으므로, 전부 생략하면
    status 필터만 걸린 기본 목록이 된다. 필터 값 검증은 호출자(포탈 API) 책임이고, ``date_col``
    만 이 함수가 검증한다.

    Args:
        status: 조회할 엣지 상태. ``proposed``(검토 큐)·``active``(승인)·``rejected``(반려).
        limit: 페이지 크기.
        offset: 건너뛸 행 수.
        q: 통합 검색어(엣지 id·양끝 자산 id·경로·사유·주제에 부분 일치).
        asset_id: 이 자산이 양끝 중 하나인 엣지만.
        kind_code: 관계 종류 필터.
        modality: 양끝 중 하나의 모달리티 필터.
        min_confidence: 신뢰도 하한(이상).
        max_confidence: 신뢰도 상한(이하).
        reviewed_by: 검토자 필터.
        since: 기간 시작(포함).
        until: 기간 끝(미포함).
        date_col: 기간 기준 컬럼(``created_at``|``reviewed_at``).

    Returns:
        ``{rows, total, status, limit, offset}``. ``total`` 은 **같은 필터**로 센 전체 건수라
        페이징 UI 의 쪽수와 목록이 어긋나지 않는다. ``rows`` 는 신뢰도 내림차순(동점은 edge_id).

    Raises:
        ValueError: ``date_col`` 이 허용 목록 밖일 때(``_build_review_where`` 가 던진다).
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

    Args:
        r: ``list_edges_for_review`` 의 조회 행(dict_row).

    Returns:
        UI 용 dict. ``src``/``dst`` 각각 ``{asset_id, file_name, modality}`` 를 갖는다.
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
    """관계 종류 목록을 조회한다(필터 드롭다운용·조회 전용).

    설명(``description``)까지 함께 읽는 이유: 드롭다운에서 "이 관계가 무슨 뜻인지" 보여주려면
    코드만으로는 부족하다. 정렬을 고정해 목록 순서가 매번 같게 한다.

    Args:
        status: 상태 필터(``active``|``inactive``). ``None`` 이면 **전체**를 돌려준다.

    Returns:
        ``{rows, total}``. rows 는 ``kind_code`` 오름차순(결정적). ``description`` 은 nullable
        이라 ``None`` 일 수 있다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        if status is not None:
            cur.execute(
                "SELECT kind_code, kind_name_ko, description, status FROM relation_kind"
                " WHERE status = %s ORDER BY kind_code",
                (status,),
            )
        else:
            cur.execute(
                "SELECT kind_code, kind_name_ko, description, status FROM relation_kind"
                " ORDER BY kind_code"
            )
        rows = [dict(r) for r in cur.fetchall()]
    return {"rows": rows, "total": len(rows)}


# 069 T406: ``list_proposed_edges`` 를 삭제했다 — 유일 소비였던 ``run_relations_review`` CLI 가
# 052 포탈 API(``GET /admin/relations`` = ``list_edges_for_review``)로 상위호환 대체되며 함께
# 폐기됐다(사용자 결정 2026-07-20). proposed 큐 조회는 ``list_edges_for_review(status="proposed")``
# 가 식별보강·페이징·검색/필터까지 포함해 담당한다.


def _decide_edge(conn: Connection[Any], *, edge_id: str, reviewer: str, status: str) -> bool:
    """proposed 엣지만 status 로 확정한다 — 이미 결정된 엣지는 건드리지 않는다.

    **DB에 쓴다**(UPDATE). 커밋은 호출자 몫이다.

    Args:
        edge_id: 대상 엣지.
        reviewer: 결정을 내린 사람(``reviewed_by`` 에 기록).
        status: 확정할 상태(``active`` 또는 ``rejected``).

    Returns:
        1행을 실제로 바꿨으면 True. **False 는 두 경우를 함께 뜻한다** — 엣지가 없거나, 이미
        결정돼 proposed 가 아니거나. 호출자는 이 값으로 "내 결정이 반영됐는지"만 판단한다.

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

    **DB에 쓴다**.

    Args:
        edge_id: 승인할 엣지.
        reviewer: 승인자.

    Returns:
        실제로 승인했으면 True. 이미 결정됐거나 없으면 False(``_decide_edge`` 계약과 동일).
    """
    return _decide_edge(conn, edge_id=edge_id, reviewer=reviewer, status="active")


def reject_edge(conn: Connection[Any], *, edge_id: str, reviewer: str) -> bool:
    """proposed 엣지를 rejected 로 반려 — **소프트**(삭제 아님, status 전이만)라 status 필터에서 빠질 뿐 행은 남는다.

    이 rejected 결정은 사람의 판단이므로, 이후 LLM 이 같은 쌍을 재제안해도
    ``sync_graph_edges`` 의 ON CONFLICT 가 status 를 덮지 않아 rejected 가 보존된다(부활 방지).

    **DB에 쓴다**.

    Args:
        edge_id: 반려할 엣지.
        reviewer: 반려자.

    Returns:
        실제로 반려했으면 True. 이미 결정됐거나 없으면 False.
    """
    return _decide_edge(conn, edge_id=edge_id, reviewer=reviewer, status="rejected")


def bulk_review(
    conn: Connection[Any], *, edge_ids: list[str], reviewer: str, action: str
) -> list[dict[str, Any]]:
    """여러 건을 한 번에 승인·반려한다 — 건별로 단건 함수를 부르고 결과를 모은다.

    로직을 다시 구현하지 않고 단건 함수에 위임하므로 가드·멱등성이 그대로 유지된다.
    커밋은 호출자가 한다 — 성공한 건들은 **한 트랜잭션에서 함께** 반영된다.

    Args:
        edge_ids: 처리할 엣지 id 목록. 순서대로 처리하며 결과도 같은 순서로 돌려준다.
        reviewer: 결정자(모든 건에 동일하게 기록).
        action: ``"approve"`` 또는 ``"reject"``. **다른 값은 즉시 예외** — 오타가 조용히 반려로
            처리되는 사고를 막는다.

    Returns:
        ``[{"edge_id", "ok"}]``. ``ok=False`` 는 그 건이 없거나 이미 결정됐다는 뜻이며, 예외가
        아니라 결과값이므로 뒤 건들의 처리를 멈추지 않는다.

    Raises:
        ValueError: ``action`` 이 허용 값이 아닐 때.
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
    """이미 내린 결정을 사람이 **되돌리는** 유일한 경로 — proposed 가드를 우회한다.

    다른 결정 함수들은 "아직 검토 전인 엣지"만 바꾸도록 막혀 있다. 그러면 잘못 승인·반려한 것을
    되돌릴 수 없으므로, 이 함수만 그 조건 없이 상태를 전이한다.

    사람이 정정한 값은 LLM 재제안에 덮이지 않는다 — 저장 경로가 status 를 갱신하지 않기 때문이다.

    **DB에 쓴다**.

    Args:
        edge_id: 정정할 엣지.
        reviewer: 정정한 사람.
        to_status: 바꿀 상태. **화이트리스트 검증은 호출자 책임**이다(포탈 API 가
            ``_REVIEW_STATUSES`` 로 막는다) — 이 함수는 받은 값을 그대로 쓴다.

    Returns:
        1행을 바꿨으면 True, 대상 엣지가 없으면 False.
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
    """LLM 제안으로 쌓인 inactive 관계 종류를 active 로 승격한다(어휘 거버넌스).

    **DB에 쓴다**. 승격된 뒤에야 그 종류가 실제 엣지로 저장될 수 있다.

    Args:
        kind_code: 승격할 관계 종류 코드.
        reviewer: 승격한 사람. **현재는 저장되지 않는다**(아래 "reviewer 미저장 이유").

    Returns:
        실제로 승격했으면 True. 이미 active 이거나 없는 코드면 False(멱등).

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
