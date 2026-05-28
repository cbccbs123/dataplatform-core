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
    """LLM 프롬프트에 노출할 **active** relation_kind 목록(레거시 코드 제외).

    레거시 제외 이유
        ``LEGACY_DOMAIN_TYPE_CODES``(medical·computer 등)는 과거 MVP에서 도메인을 kind_code로
        쓰던 잔재다. 이 값들을 프롬프트에 포함하면 LLM이 도메인을 관계 종류로 혼용해 제안한다.
        prompt.py 가 이 결과를 그대로 카탈로그 블록에 넣으므로 여기서 배제해야 한다.
    """
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
    """``kind_code`` 로 relation_kind 행(id, is_symmetric) 조회. ``status`` None 이면 상태 무시.

    주의: 기본값 ``status='active'`` 는 graph_persist 가 엣지를 확정할 때 **활성 종류만** 허용한다는
    불변식을 강제한다. inactive kind 를 엣지화하려면 명시적으로 ``status=None`` 을 전달해야 한다.
    """
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
    """LLM 제안 신규 kind_code 를 relation_kind 에 등록(기본 inactive=검토 전). relation_kind_id(str) 반환.

    멱등 보장(ON CONFLICT DO UPDATE)
        이미 같은 kind_code 가 존재하면 INSERT 는 실패하고 UPDATE 로 전환된다.
        이때 ``COALESCE(NULLIF(EXCLUDED.…, ''), relation_kind.…)`` 패턴은
        "신규 값이 빈 문자열이 아닐 때만 덮어쓰고, 비면 기존 값 보존"을 의미한다.
        즉 같은 코드를 LLM 이 여러 번 제안해도 **기존 이름·설명이 소실되지 않는다**.

    status 미변경 이유
        ON CONFLICT DO UPDATE 에 status 를 포함하지 않는다.
        이미 active 로 승격된 kind 가 다시 LLM 제안을 받아도 inactive 로 강등되는 사고를 방지한다.

    RETURNING 실패 fallback
        PostgreSQL 의 ON CONFLICT DO UPDATE 는 UPDATE 가 발생해도 RETURNING 을 반환하지만,
        충돌 행이 트리거 등으로 실제로 변경되지 않으면 드물게 None 이 올 수 있다.
        이 경우 SELECT 로 재조회해 id 를 보장한다.

    PK 생성: ``gen_random_uuid()`` 는 DB 측 생성이므로 앱 UUID(UUIDv7) 규칙 예외.
        relation_kind 는 정적 시드 테이블이라 생성 순서 보장이 불필요하다.
    """
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
        # RETURNING 이 None 인 드문 경우(트리거·행 변경 없음): 재조회로 보장
        cur.execute("SELECT relation_kind_id FROM relation_kind WHERE kind_code = %s LIMIT 1", (kind_code,))
        row2 = cur.fetchone()
        if row2 is None:
            raise RuntimeError("ensure_relation_kind_for_llm_proposal: relation_kind_id 해소 실패")
        return str(row2["relation_kind_id"])
