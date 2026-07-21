"""F-4.5 PROV-DM 처리 이력 기록 — ``asset_lineage`` 한 행 INSERT.

오케스트레이터가 단계별 트랜잭션 안에서 호출한다(``conn`` 우선 인자 — 기존 ``registry/*_persist.py`` 패턴).
PROV-DM(W3C 2013) Entity-Activity-Agent 모델: ``activity``(무엇이 일어났나) · ``agent``(누가/무엇이) ·
``used``/``generated``/``payload``(jsonb 부가). PK 는 앱 생성 UUIDv7.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg import Connection

from src.database.ids import uuid7


def record_lineage(
    conn: Connection[Any],
    asset_id: uuid.UUID | str,
    *,
    activity: str,
    agent: str,
    used: dict[str, Any] | None = None,
    generated: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> uuid.UUID:
    """``asset_lineage`` 에 활동 1건을 기록하고 lineage_id 반환. 호출자가 트랜잭션 경계를 제어.

    PROV-DM(W3C 2013) 매핑:
    - ``activity``: 일어난 처리 단계 이름 (예: "extract_text", "embed_text")
    - ``agent``    : 수행 주체 식별자 (예: "run_ingest", "dispatcher")
    - ``used``     : 입력 자원 기술 (파일 경로·체크섬 등)
    - ``generated``: 출력 자원 기술 (메타 키 목록, 임베딩 채널 등)
    - ``payload``  : 위에 맞지 않는 부가 정보 (자유형 jsonb)

    ``occurred_at`` 은 DB 서버 ``now()`` 로 기록된다(DDL DEFAULT). 앱 시계와 무관해
    시간대·NTP 편차 없이 일관된 순서를 보장한다.
    """
    lineage_id = uuid7()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset_lineage (lineage_id, asset_id, activity, agent, used, generated, payload)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
            """,
            (
                lineage_id,
                asset_id,
                activity,
                agent,
                json.dumps(used or {}, ensure_ascii=False),
                json.dumps(generated or {}, ensure_ascii=False),
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )
    return lineage_id
