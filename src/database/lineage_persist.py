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

    ``occurred_at`` 은 앱이 아니라 **DB 서버 시계**로 찍힌다 — 여러 워커가 각자 시계로 찍으면
    시간대·시각 오차 때문에 순서가 뒤엉킨다.

    **DB에 쓴다**(INSERT 1건). 커밋은 호출자 몫이다.

    Args:
        asset_id: 활동 대상 자산.
        activity: 무슨 처리였는지(예: ``extract_text``·``relations.proposed.v1``).
        agent: 누가 했는지(실행 주체 이름).
        used: 입력 자원 기술. ``None`` 이면 빈 객체로 저장한다.
        generated: 산출 기술. **결정적으로 정렬해 넣을 것** — 같은 입력이면 계보도 같아야 비교가 된다.
        payload: 위 셋에 맞지 않는 부가 정보.

    Returns:
        새로 만든 ``lineage_id``.
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
