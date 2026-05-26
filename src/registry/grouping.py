"""F-1.3 자산 그룹화.

study_uid / MRN 등 그룹 키로 ``asset_group`` 을 upsert 하고 ``group_id`` 를 돌려준다.
일반 도메인(현재 데이터)은 그룹 키가 없어 ``None`` (그룹 없음). 의료 도메인에서 study_uid/MRN
기반 묶음에 사용(후속 단계에서 core_meta 로부터 키 추출).
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

# core_meta 에서 그룹 키로 시도할 (group_kind, meta_key) 우선순위.
_GROUP_KEY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("study_uid", "study_uid"),
    ("mrn", "mrn"),
)


def upsert_group(conn: Connection[Any], *, group_kind: str, group_key: str) -> int:
    """(group_kind, group_key) 로 ``asset_group`` 을 보장하고 ``group_id`` 반환."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            INSERT INTO asset_group (group_kind, group_key)
            VALUES (%s, %s)
            ON CONFLICT (group_kind, group_key) DO UPDATE SET group_kind = EXCLUDED.group_kind
            RETURNING group_id
            """,
            (group_kind, group_key),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("asset_group upsert 가 group_id 를 반환하지 않았습니다.")
    return int(row["group_id"])


def resolve_group_id(conn: Connection[Any], core_meta: dict[str, Any]) -> int | None:
    """core_meta 에서 그룹 키(study_uid/mrn)를 찾아 ``asset_group`` upsert. 없으면 None."""
    for group_kind, meta_key in _GROUP_KEY_CANDIDATES:
        val = core_meta.get(meta_key)
        if val is not None and str(val).strip():
            return upsert_group(conn, group_kind=group_kind, group_key=str(val).strip())
    return None
