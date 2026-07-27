"""**관계 종류(relation_kind) 카탈로그** 조회·보장.

엣지는 ``relation_kind``(통제 어휘)를 직접 참조하고, 주제 라벨은 ``graph_edge.topic`` jsonb 에 산다.
LLM이 새 관계 종류를 제안하면 **inactive 로만** 등록되고, 사람이 승인해야 엣지에 쓰인다.
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

    Returns:
        ``{type_code, type_name, description, is_symmetric}`` 행 리스트. ``kind_code`` 오름차순
        고정(결정적). 활성 종류가 없으면 빈 리스트.
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
    """``kind_code`` 로 relation_kind 한 행을 조회한다.

    Args:
        kind_code: 관계 종류 코드(소문자 정규화된 값).
        status: 상태 필터. 기본 ``'active'`` 는 **활성 종류만 엣지가 될 수 있다**는 불변식을
            강제한다(``graph_persist`` 가 이 기본값에 의존). ``None`` 이면 **상태를 보지 않고**
            찾는다 — inactive kind 까지 필요할 때만 명시적으로 넘긴다.

    Returns:
        ``{relation_kind_id, is_symmetric}``. 조건에 맞는 행이 없으면 ``None``.
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
    """LLM이 제안한 새 관계 종류를 ``relation_kind`` 에 등록한다(기본 inactive = 검토 전).

    **DB에 쓴다**(INSERT 또는 UPDATE). 호출자의 트랜잭션 안에서 돈다.

    Args:
        kind_code: 관계 종류 코드. 이 값이 충돌 기준(유니크 키)이다.
        kind_name_ko: 한국어 표시 이름. 비면 ``kind_code`` 를 쓰고 255자로 자른다.
        description: 설명. 비면 "LLM 제안으로 자동 등록됨."이 들어간다.
        is_symmetric: 방향이 없는 관계인지(A-B = B-A). **엣지 저장 순서를 좌우한다**
            (``_canonical_pair``) — True 면 같은 쌍이 한 행으로 모인다.
        status: 등록 상태. 기본 ``'inactive'``(검토 대기)를 유지해야 한다 — LLM 제안이 사람 승인
            없이 곧바로 그래프에 반영되는 것을 막는 안전장치다.

    Returns:
        확정된 ``relation_kind_id``(문자열).

    Raises:
        RuntimeError: INSERT·UPDATE·재조회가 모두 id 를 못 준 경우(정상 경로에선 발생하지 않음).

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
