"""F-5.1 분류 결과 영속화.

``asset_classification`` 적재 + ``asset.domain_label``/``domain_confidence`` 갱신.
final_label == 'review' 면 ``unresolved_pool`` 에 등록(HITL 대상).
psycopg ``Connection`` 을 받아 오케스트레이터 트랜잭션에서 조합한다.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg import Connection

from src.classify.types import DOMAIN_REVIEW, ClassificationResult
from src.database.ids import uuid7

# unresolved_pool.reason 값 — run_ingest 팩 선택 로직이 이 reason 으로 HITL 대상을 식별한다.
_REASON_CLASSIFICATION_REVIEW = "classification_review"


def record_classification(conn: Connection[Any], asset_id: uuid.UUID, result: ClassificationResult) -> None:
    """분류 결과를 asset_classification 에 적재하고 asset 도메인 라벨을 갱신한다.

    재분류(동일 asset_id 재실행) 시 ON CONFLICT 로 최신 결과로 덮어쓴다.
    stage*_scores 는 cascade 각 단계(시그니처/제로샷/LLM)가 산출한 도메인별 점수 jsonb 이며,
    ``decided_stage`` 가 실제로 최종 판정을 내린 단계다. 이후 팩 선택·deferred 판별의 기반.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset_classification (
                classification_id, asset_id, stage1_scores, stage2_scores, stage3_scores,
                final_label, confidence, decided_stage, policy_version
            ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (asset_id) DO UPDATE SET
                stage1_scores = EXCLUDED.stage1_scores,
                stage2_scores = EXCLUDED.stage2_scores,
                stage3_scores = EXCLUDED.stage3_scores,
                final_label   = EXCLUDED.final_label,
                confidence    = EXCLUDED.confidence,
                decided_stage = EXCLUDED.decided_stage,
                policy_version = EXCLUDED.policy_version
            """,
            (
                uuid7(), asset_id,
                json.dumps(result.stage1_scores, ensure_ascii=False),
                json.dumps(result.stage2_scores, ensure_ascii=False),
                json.dumps(result.stage3_scores, ensure_ascii=False),
                result.final_label, result.confidence, result.decided_stage, result.policy_version,
            ),
        )
        # asset 테이블도 동기화 — run_ingest 가 domain_label 로 팩 선택·deferred 판별
        cur.execute(
            "UPDATE asset SET domain_label = %s, domain_confidence = %s, updated_at = now() WHERE asset_id = %s",
            (result.final_label, result.confidence, asset_id),
        )
        if result.final_label == DOMAIN_REVIEW:
            # 'review' 판정 자산은 HITL 대기열(unresolved_pool) 에 등록.
            # ON CONFLICT DO NOTHING: 중복 실행해도 풀 항목은 한 번만 생성된다.
            cur.execute(
                """
                INSERT INTO unresolved_pool (pool_id, asset_id, reason, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (asset_id) DO NOTHING
                """,
                (uuid7(), asset_id, _REASON_CLASSIFICATION_REVIEW, json.dumps({"decided_stage": result.decided_stage})),
            )
