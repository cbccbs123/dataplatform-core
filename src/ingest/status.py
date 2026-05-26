"""F-1.4 자산 처리 상태 머신.

``asset.status`` 전이 규칙과 DB 갱신 헬퍼. 모델 A(조기 INSERT) 전제:
오케스트레이터가 파일 픽업 시 ``asset`` 행을 ``received`` 로 만들고,
단계가 진행될 때마다 ``set_status`` 로 전이한다. 임의 단계에서 실패하면
``mark_failed`` 로 ``failed`` + 사유를 남기고 다음 파일로 넘어간다.

디스패처 단일 권위: 미지원 modality 등은 사전 차단하지 않고 흐름에 태운 뒤,
``dispatch_extract`` 의 예외를 오케스트레이터가 잡아 ``mark_failed`` 로 흡수한다.

상태 헬퍼는 psycopg ``Connection`` 을 받아 오케스트레이터의 트랜잭션 안에서 조합된다
(``src/relations/*`` 와 동일 패턴). 전이 검증 로직(``validate_transition``)은 DB 없이도 순수 호출 가능.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row


class AssetStatus(str, Enum):
    """``asset.status`` CHECK 제약과 동일한 값."""

    RECEIVED = "received"
    ROUTING = "routing"
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    REGISTERED = "registered"
    FAILED = "failed"
    DEFERRED = "deferred"  # 의료 표준 포맷(DICOM/HL7/FHIR) 추출 보류 — 실패 아님


# 종료 상태(더 이상 전이 없음).
TERMINAL: frozenset[AssetStatus] = frozenset(
    {AssetStatus.REGISTERED, AssetStatus.FAILED, AssetStatus.DEFERRED}
)

# 정상 진행 경로 + 임의 비종료 단계에서 failed 로 전이 가능. classifying 에서 deferred(추출 보류) 가능.
ALLOWED_TRANSITIONS: dict[AssetStatus, frozenset[AssetStatus]] = {
    AssetStatus.RECEIVED: frozenset({AssetStatus.ROUTING, AssetStatus.FAILED}),
    AssetStatus.ROUTING: frozenset({AssetStatus.CLASSIFYING, AssetStatus.FAILED}),
    AssetStatus.CLASSIFYING: frozenset({AssetStatus.EXTRACTING, AssetStatus.DEFERRED, AssetStatus.FAILED}),
    AssetStatus.EXTRACTING: frozenset({AssetStatus.REGISTERED, AssetStatus.FAILED}),
    AssetStatus.REGISTERED: frozenset(),
    AssetStatus.FAILED: frozenset(),
    AssetStatus.DEFERRED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    """허용되지 않은 상태 전이."""


def validate_transition(current: AssetStatus | str, target: AssetStatus | str) -> None:
    """``current → target`` 이 허용 전이가 아니면 ``InvalidTransitionError``."""
    cur = AssetStatus(current)
    tgt = AssetStatus(target)
    if tgt not in ALLOWED_TRANSITIONS.get(cur, frozenset()):
        raise InvalidTransitionError(f"{cur.value} → {tgt.value} 전이는 허용되지 않습니다.")


def fetch_status(conn: Connection[Any], asset_id: uuid.UUID) -> AssetStatus:
    """``asset.status`` 조회. 없으면 ``LookupError``."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT status FROM asset WHERE asset_id = %s", (asset_id,))
        row = cur.fetchone()
    if row is None:
        raise LookupError(f"asset 없음: asset_id={asset_id}")
    return AssetStatus(row["status"])


def set_status(
    conn: Connection[Any],
    asset_id: uuid.UUID,
    target: AssetStatus | str,
    *,
    reason: str | None = None,
) -> None:
    """현재 상태를 읽어 전이를 검증한 뒤 ``asset.status`` 를 갱신한다.

    ``reason`` 은 ``status_reason`` 에 기록(정상 전이 시 None → 이전 사유 클리어).
    """
    tgt = AssetStatus(target)
    current = fetch_status(conn, asset_id)
    validate_transition(current, tgt)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE asset SET status = %s, status_reason = %s, updated_at = now() WHERE asset_id = %s",
            (tgt.value, reason, asset_id),
        )


def mark_failed(conn: Connection[Any], asset_id: uuid.UUID, reason: str) -> None:
    """현재 단계에서 ``failed`` 로 전이하고 사유를 남긴다(디스패처 예외 등 흡수용)."""
    set_status(conn, asset_id, AssetStatus.FAILED, reason=reason)
