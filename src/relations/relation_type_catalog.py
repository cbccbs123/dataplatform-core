"""**관계 종류(relation_kind) 카탈로그** 조회·보장.

C+ 슬림화 이후 엣지는 ``relation_kind`` 를 직접 참조하고 주제는 ``graph_edge.topic`` jsonb 에 산다.
``relation_type``/``relation_subtopic``/``relation_topic_parent`` 는 v230 에서 드롭됐다.
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.relations.schema import LEGACY_DOMAIN_TYPE_CODES


def fetch_active_relation_kinds(conn: Connection[Any]) -> list[dict[str, Any]]:
    """LLM 프롬프트에 노출할 **active** relation_kind 목록(레거시 코드 제외)."""
    legacy = list(LEGACY_DOMAIN_TYPE_CODES)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT kind_code AS type_code,
                   COALESCE(kind_name_ko, '') AS type_name,
                   COALESCE(description, '') AS description,
                   is_symmetric
            FROM relation_kind
            WHERE status = 'active'
              AND kind_code <> ALL(%s::text[])
            ORDER BY kind_code
            """,
            (legacy,),
        )
        return [dict(r) for r in cur.fetchall()]


def fetch_relation_kind(conn: Connection[Any], *, kind_code: str, status: str | None = "active") -> dict[str, Any] | None:
    """``kind_code`` 로 relation_kind 행(id, is_symmetric) 조회. ``status`` None 이면 상태 무시."""
    q = "SELECT relation_kind_id, is_symmetric FROM relation_kind WHERE kind_code = %s"
    params: list[Any] = [kind_code]
    if status is not None:
        q += " AND status = %s"
        params.append(status)
    q += " LIMIT 1"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(q, tuple(params))
        row = cur.fetchone()
    return dict(row) if row else None


def ensure_relation_kind_for_llm_proposal(
    conn: Connection[Any],
    *,
    kind_code: str,
    kind_name_ko: str,
    description: str,
    is_symmetric: bool = True,
    status: str = "inactive",
) -> str:
    """LLM 제안 신규 kind_code 를 relation_kind 에 등록(기본 inactive=검토 전). relation_kind_id(str) 반환."""
    kn = (kind_name_ko or kind_code).strip()[:255] or kind_code
    desc = (description or "").strip() or "LLM 제안으로 자동 등록됨."
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO relation_kind (relation_kind_id, kind_code, kind_name_ko, description, is_symmetric, status)
            VALUES (gen_random_uuid(), %s, %s, %s, %s, %s)
            ON CONFLICT (kind_code) DO UPDATE SET
                kind_name_ko = COALESCE(NULLIF(EXCLUDED.kind_name_ko, ''), relation_kind.kind_name_ko),
                description  = COALESCE(NULLIF(EXCLUDED.description, ''), relation_kind.description)
            RETURNING relation_kind_id
            """,
            (kind_code, kn, desc, is_symmetric, status),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row["relation_kind_id"])
        cur.execute("SELECT relation_kind_id FROM relation_kind WHERE kind_code = %s LIMIT 1", (kind_code,))
        row2 = cur.fetchone()
        if row2 is None:
            raise RuntimeError("ensure_relation_kind_for_llm_proposal: relation_kind_id 해소 실패")
        return str(row2["relation_kind_id"])
