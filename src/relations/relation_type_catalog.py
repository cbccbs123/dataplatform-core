"""**관계 카탈로그** 테이블 조회·보장: ``relation_kind``, ``relation_topic_parent``, ``relation_subtopic``, ``relation_type``.

스키마 요약
    - ``relation_topic_parent``: 대주제 (``topic_ko``, ``topic_en``) 유니크.
    - ``relation_subtopic``: 세부 주제 (``topic_id``, ``subtopic_ko``, ``subtopic_en``) 유니크.
    - ``relation_type``: (``relation_kind_id``, ``relation_subtopic_id``) 조합 한 행. ``status`` 가 ``active`` 인 것만
      LLM 프롬프트 카탈로그에 실린다.

시퀀스
    ``relation_type`` 갱신은 ``INSERT … ON CONFLICT`` 대신 **SELECT FOR UPDATE → UPDATE 또는 INSERT** 로
    동일 (kind, subtopic) 반복 시 ``relation_type_id`` 시퀀스 낭비를 줄인다.

서브토픽 행 폭주(선택 설계, 미구현)
    - **허용 집합**: 토픽별 ``allowed_subtopic`` 테이블 + 적재 시 매핑.
    - **스테이징**: ``relation_subtopic_candidate`` 에만 적재 후 운영 머지.
    현재는 ``schema.normalize_subtopic_*``(첫 어절·첫 토큰)과 프롬프트 규칙으로 1차 완화한다.
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.relations.schema import LEGACY_DOMAIN_TYPE_CODES

# relation_type ↔ kind ↔ topic/subtopic 조인 (프롬프트·조회 공통)
_RT_JOIN = """
    FROM relation_type rt
    JOIN relation_kind rk ON rk.relation_kind_id = rt.relation_kind_id
    JOIN relation_subtopic s ON s.subtopic_id = rt.relation_subtopic_id
    JOIN relation_topic_parent tp ON tp.topic_id = s.topic_id
"""


def fetch_llm_prompt_relation_types(conn: Connection[Any]) -> list[dict[str, Any]]:
    """
    LLM **관계 제안** 프롬프트에 넣을 ``relation_type`` 목록.

    - ``rt.status = 'active'`` 만: 비활성 번들은 LLM이 코드를 골라도 안 되게 숨긴다.
    - ``type_name`` / ``description`` 은 ``relation_type`` 컬럼이 아니라 **``relation_kind``** 에서 가져온다.
    - ``LEGACY_DOMAIN_TYPE_CODES`` 는 과거 도메인-as-code 패턴이라 제외.
    """
    legacy = list(LEGACY_DOMAIN_TYPE_CODES)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT rk.kind_code AS type_code,
                   COALESCE(rk.kind_name_ko, '') AS type_name,
                   COALESCE(tp.topic_ko, '') AS topic_ko,
                   COALESCE(s.subtopic_ko, '') AS subtopic_ko,
                   COALESCE(tp.topic_en, '') AS topic_en,
                   COALESCE(s.subtopic_en, '') AS subtopic_en,
                   COALESCE(rk.description, '') AS description,
                   rt.relation_type_id
            {_RT_JOIN}
            WHERE rt.status = 'active'
              AND rk.kind_code <> ALL(%s::text[])
            ORDER BY rk.kind_code, tp.topic_ko, tp.topic_en
            """,
            (legacy,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def fetch_relation_types_for_prompt_dedup(conn: Connection[Any]) -> list[dict[str, Any]]:
    """
    프롬프트 **dedup(기존 등록)** 블록용: ``active``·``inactive`` ``relation_type`` 모두.

    LLM이 이미 DB에 **비활성으로만** 등록된 (kind, topic) 번들과 같은 코드를 다시 쓰지 않도록
    ``status`` 필드를 JSON에 포함한다.
    """
    legacy = list(LEGACY_DOMAIN_TYPE_CODES)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT rk.kind_code AS type_code,
                   COALESCE(rk.kind_name_ko, '') AS type_name,
                   COALESCE(tp.topic_ko, '') AS topic_ko,
                   COALESCE(s.subtopic_ko, '') AS subtopic_ko,
                   COALESCE(tp.topic_en, '') AS topic_en,
                   COALESCE(s.subtopic_en, '') AS subtopic_en,
                   COALESCE(rk.description, '') AS description,
                   rt.status,
                   rt.relation_type_id
            {_RT_JOIN}
            WHERE rk.kind_code <> ALL(%s::text[])
            ORDER BY CASE WHEN rt.status = 'active' THEN 0 ELSE 1 END,
                     rk.kind_code ASC,
                     tp.topic_ko,
                     tp.topic_en
            """,
            (legacy,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def fetch_relation_kind_id(conn: Connection[Any], *, kind_code: str) -> str | None:
    """``kind_code`` 로 ``relation_kind_id``(UUID str) 조회. 없으면 None."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT relation_kind_id
            FROM relation_kind
            WHERE kind_code = %s
            LIMIT 1
            """,
            (kind_code,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return str(row["relation_kind_id"])


def ensure_relation_kind_for_llm_proposal(
    conn: Connection[Any],
    *,
    kind_code: str,
    kind_name_ko: str,
    description: str,
    is_symmetric: bool = True,
    status: str = "inactive",
) -> str:
    """
    LLM이 제안한 **새 kind_code** 를 ``relation_kind`` 에 반영(또는 기존 행 보강). ``relation_kind_id``(UUID str) 반환.

    - 신규: ``INSERT`` (PK 는 ``gen_random_uuid()``, 보통 ``status='inactive'`` — 검토 전).
    - 기존: ``ON CONFLICT`` 로 **이름·설명만** 갱신; ``status`` 는 덮어쓰지 않는다(이미 active 면 유지).
    """
    kn = (kind_name_ko or kind_code).strip() or kind_code
    if len(kn) > 255:
        kn = kn[:255]
    desc = (description or "").strip() or "LLM 제안으로 자동 등록됨."
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO relation_kind (relation_kind_id, kind_code, kind_name_ko, description, is_symmetric, status)
            VALUES (gen_random_uuid(), %s, %s, %s, %s, %s)
            ON CONFLICT (kind_code) DO UPDATE SET
                kind_name_ko = COALESCE(NULLIF(EXCLUDED.kind_name_ko, ''), relation_kind.kind_name_ko),
                description = COALESCE(NULLIF(EXCLUDED.description, ''), relation_kind.description)
            RETURNING relation_kind_id
            """,
            (kind_code, kn, desc, is_symmetric, status),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row["relation_kind_id"])
        cur.execute(
            """
            SELECT relation_kind_id FROM relation_kind WHERE kind_code = %s LIMIT 1
            """,
            (kind_code,),
        )
        row2 = cur.fetchone()
        if row2 is None:
            raise RuntimeError("ensure_relation_kind_for_llm_proposal: failed to resolve relation_kind_id")
        return str(row2["relation_kind_id"])


def ensure_relation_topic_parent_row(
    conn: Connection[Any],
    *,
    topic_ko: str,
    topic_en: str,
) -> str:
    """
    대주제 (``topic_ko``, ``topic_en``) 행을 보장하고 ``topic_id``(UUID str) 를 반환한다.
    """
    tk = (topic_ko or "").strip()
    te = (topic_en or "").strip()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO relation_topic_parent (topic_id, topic_ko, topic_en)
            VALUES (gen_random_uuid(), %s, %s)
            ON CONFLICT (topic_ko, topic_en) DO NOTHING
            RETURNING topic_id
            """,
            (tk, te),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row["topic_id"])
        cur.execute(
            """
            SELECT topic_id
            FROM relation_topic_parent
            WHERE topic_ko = %s AND topic_en = %s
            LIMIT 1
            """,
            (tk, te),
        )
        row2 = cur.fetchone()
        if row2 is None:
            raise RuntimeError("ensure_relation_topic_parent_row: failed to resolve topic_id")
        return str(row2["topic_id"])


def ensure_relation_subtopic_row(
    conn: Connection[Any],
    *,
    topic_id: str,
    subtopic_ko: str,
    subtopic_en: str,
) -> str:
    """
    ``topic_id`` 아래 (``subtopic_ko``, ``subtopic_en``) 행을 보장하고 ``subtopic_id``(UUID str) 를 반환한다.
    """
    sk = (subtopic_ko or "").strip()
    se = (subtopic_en or "").strip()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO relation_subtopic (subtopic_id, topic_id, subtopic_ko, subtopic_en)
            VALUES (gen_random_uuid(), %s, %s, %s)
            ON CONFLICT (topic_id, subtopic_ko, subtopic_en) DO NOTHING
            RETURNING subtopic_id
            """,
            (topic_id, sk, se),
        )
        row = cur.fetchone()
        if row is not None:
            return str(row["subtopic_id"])
        cur.execute(
            """
            SELECT subtopic_id
            FROM relation_subtopic
            WHERE topic_id = %s AND subtopic_ko = %s AND subtopic_en = %s
            LIMIT 1
            """,
            (topic_id, sk, se),
        )
        row2 = cur.fetchone()
        if row2 is None:
            raise RuntimeError("ensure_relation_subtopic_row: failed to resolve subtopic_id")
        return str(row2["subtopic_id"])


def ensure_relation_subtopic_bundle(
    conn: Connection[Any],
    *,
    topic_ko: str,
    subtopic_ko: str,
    topic_en: str,
    subtopic_en: str,
) -> str:
    """
    LLM 토피 네 필드에 대응하는 ``relation_subtopic.subtopic_id``(UUID str) 를 보장·반환한다.

    내부적으로 ``relation_topic_parent`` + ``relation_subtopic`` 순으로 upsert 한다.
    """
    tid = ensure_relation_topic_parent_row(conn, topic_ko=topic_ko, topic_en=topic_en)
    return ensure_relation_subtopic_row(
        conn,
        topic_id=tid,
        subtopic_ko=subtopic_ko,
        subtopic_en=subtopic_en,
    )


def fetch_relation_type_id(
    conn: Connection[Any],
    *,
    relation_kind_id: str,
    relation_subtopic_id: str,
    status: str | None = "active",
) -> str | None:
    """
    (``relation_kind_id``, ``relation_subtopic_id``) 에 해당하는 ``relation_type_id``(UUID str).

    ``status``:
        - ``'active'``: 확정 적재 직전 조회에 사용.
        - ``None``: 상태 무시(비활성 번들 조회 등).
    """
    q = """
            SELECT relation_type_id
            FROM relation_type
            WHERE relation_kind_id = %s
              AND relation_subtopic_id = %s
        """
    params: list[Any] = [relation_kind_id, relation_subtopic_id]
    if status is not None:
        q += " AND status = %s"
        params.append(status)
    q += " LIMIT 1"
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(q, tuple(params))
        row = cur.fetchone()
    if row is None:
        return None
    return str(row["relation_type_id"])


def fetch_relation_type_id_for_normalized_edge(
    conn: Connection[Any],
    *,
    kind_code: str,
    topic_ko: str,
    subtopic_ko: str,
    topic_en: str,
    subtopic_en: str,
) -> str | None:
    """
    정규화된 토피 번들 + ``kind_code`` 로 ``relation_type_id``(UUID str) 를 찾는다.

    ``relation_subtopic`` 번들은 ``ensure_relation_subtopic_bundle`` 으로 보장한 뒤,
    ``relation_type`` 은 **status 무시**로 조회한다(LLM 경로에서 inactive 행이 많음).
    ``relation_kind`` 가 없으면 None.
    """
    kid = fetch_relation_kind_id(conn, kind_code=kind_code)
    if kid is None:
        return None
    sid = ensure_relation_subtopic_bundle(
        conn,
        topic_ko=topic_ko,
        subtopic_ko=subtopic_ko,
        topic_en=topic_en,
        subtopic_en=subtopic_en,
    )
    return fetch_relation_type_id(
        conn,
        relation_kind_id=kid,
        relation_subtopic_id=sid,
        status=None,
    )


def _ensure_relation_type_status(
    conn: Connection[Any],
    *,
    relation_kind_id: str,
    relation_subtopic_id: str,
    status: str,
) -> None:
    """
    ``relation_type`` 한 행의 ``status`` 를 목표값으로 맞춘다.

    1. 동일 (kind, subtopic) 행을 ``FOR UPDATE`` 로 잠근 뒤 있으면 ``UPDATE status`` 만 수행.
    2. 없으면 ``INSERT`` 로 새 행(PK 는 ``gen_random_uuid()``).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT relation_type_id
            FROM relation_type
            WHERE relation_kind_id = %s
              AND relation_subtopic_id = %s
            FOR UPDATE
            """,
            (relation_kind_id, relation_subtopic_id),
        )
        row = cur.fetchone()
        if row is not None:
            cur.execute(
                """
                UPDATE relation_type
                SET status = %s
                WHERE relation_type_id = %s
                """,
                (status, str(row["relation_type_id"])),
            )
            return
        cur.execute(
            """
            INSERT INTO relation_type (relation_type_id, relation_kind_id, relation_subtopic_id, status)
            VALUES (gen_random_uuid(), %s, %s, %s)
            """,
            (relation_kind_id, relation_subtopic_id, status),
        )


def ensure_inactive_relation_type_row(
    conn: Connection[Any],
    *,
    relation_kind_id: str,
    relation_subtopic_id: str,
) -> None:
    """검토 큐·비허용 코드 경로에서 (kind, subtopic) 번들을 ``inactive`` 로 둔다."""
    _ensure_relation_type_status(
        conn,
        relation_kind_id=relation_kind_id,
        relation_subtopic_id=relation_subtopic_id,
        status="inactive",
    )


def ensure_active_relation_type_row(
    conn: Connection[Any],
    *,
    relation_kind_id: str,
    relation_subtopic_id: str,
) -> None:
    """(kind, subtopic) 번들을 ``active`` 로 올려 LLM 카탈로그에 노출 가능하게 한다."""
    _ensure_relation_type_status(
        conn,
        relation_kind_id=relation_kind_id,
        relation_subtopic_id=relation_subtopic_id,
        status="active",
    )
