"""DB ``CHECK (… IN (…))`` 제약과 동기화하는 닫힌 어휘(StrEnum) — spec 042.

PostgreSQL ``ENUM`` 타입이 아니라 기존 마이그레이션의 ``TEXT + CHECK IN`` 과 **코드만 동기**.
신규 값 추가 시: DDL CHECK · 본 모듈 · ``tests/test_status_vocab.py`` 동시 갱신.

OM 039~042 에서 쓰이는 어휘
    ``AccessTier``              — 040 v290/v291 ``access_tier`` 컬럼
    ``RegistryFieldStatus``     — 041 ``ext_meta_field_registry.status``
    ``GraphEdgeStatus``         — cross-asset (v230, 042에서 타입 정본화만)
    ``RelationResolutionStatus``— 관계 큐 (v250/v260, 042에서 타입 정본화만)

``AssetStatus`` 정본은 파이프라인 레포 ``processing.ingest.status``(FSM 전이) — 여기서는 CHECK 집합 교차 검증만.

주의(US-F 스코프): ``GraphEdgeStatus``/``RelationResolutionStatus`` 는 prod 코드가 값을 직접
참조하지 않고 ``tests/test_status_vocab.py`` 의 DDL 교차검증만 소비한다. 그럼에도 삭제하지 않는다
— DB CHECK 제약과 동기화되는 '닫힌 어휘의 정본'이므로 소비처가 테스트뿐이어도 스키마 계약 자체다
(US-F 죽은코드 청소에서 의도적으로 제외·069 코드리뷰 2026-07-15).
"""

from __future__ import annotations

from enum import StrEnum


class AccessTier(StrEnum):
    """``ext_meta_field_registry.access_tier`` CHECK 4값 (v290·v291 · spec 040).

    ordinal(낮→높): public < authenticated < authorized < regulated.
    read projection(042): principal clearance ≥ effective_field_tier 인 키만 응답에 포함(미달 키 omit).
    """

    PUBLIC = "public"  # 익명·비인증 principal.
    AUTHENTICATED = "authenticated"  # 필드 tier·중간 clearance.
    AUTHORIZED = "authorized"  # JWT 2-tier MVP clearance 상한(042).
    REGULATED = "regulated"  # 규제·의료 등 최고 등급 — ``domain_floor(medical)`` 바닥.


class GraphEdgeStatus(StrEnum):
    """``graph_edge.status`` CHECK (v230) — cross-asset 관계 HITL 검토 상태."""

    PROPOSED = "proposed"  # LLM 제안·미검토(기본값).
    ACTIVE = "active"  # 승인·자동승인 — graph_query·포탈 노출 대상.
    REJECTED = "rejected"  # 반려 — 검색·그래프 조회에서 제외.


class RelationResolutionStatus(StrEnum):
    """``relation_resolution.status`` CHECK (v250·v260) — 관계 해소 배치 큐 상태."""

    PENDING = "pending"  # 미해소·일시 실패 후 재시도 대기.
    RESOLVED = "resolved"  # 관계≥1 생성 완료.
    ISOLATED = "isolated"  # 평가 완료·관계 0(고립, 실패≠) — 재평가 대상.
    FAILED = "failed"  # 재시도 상한 도달(DLQ).


class RegistryFieldStatus(StrEnum):
    """``ext_meta_field_registry.status`` / 레거시 ``schema_registry.status`` (041)."""

    ACTIVE = "active"  # ingest ``validate_ext_meta`` · read ``fetch_access_tiers`` 대상.
    INACTIVE = "inactive"  # 비활성 — 레지스트리 조회·검증 제외(행 보존).
