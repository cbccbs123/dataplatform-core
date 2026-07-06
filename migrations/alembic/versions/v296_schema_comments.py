"""v296 — 스키마 컬럼 COMMENT 보강(25컬럼) + 2026-07-01 임시 백업 테이블 6개 정리

Revision ID: v296_schema_comments
Revises: v295_topic_registry_alias

실 스키마의 미기입 컬럼 COMMENT 를 채워 스키마를 자기설명적으로 만들고,
alembic 관리 밖 임시 백업(`*_bak_20260701`·relregen 전 수동 백업·FK 참조 0)을 정리한다.
DDL 본문은 migrations/sql/296_schema_comments_bak_cleanup.sql 단일 출처(데이터/제약 무변경).

downgrade: 보강한 25컬럼 COMMENT 를 IS NULL 로 원복(전부 직전 NULL). 백업 테이블 DROP 은
**비가역 정리**라 재생성하지 않는다(임시 백업 데이터는 복원 대상 아님·DROP IF EXISTS 로 재 upgrade 안전).

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v296_schema_comments"
down_revision = "v295_topic_registry_alias"
branch_labels = None
depends_on = None

# downgrade 에서 원복할 (테이블, 컬럼) — upgrade 로 새로 COMMENT 를 채운 25컬럼(직전 상태 NULL).
_COMMENTED = [
    ("ext_meta_field_registry", c)
    for c in ("domain", "meta_key", "json_schema", "description", "status", "access_tier", "created_at")
] + [
    ("graph_edge", c)
    for c in ("edge_id", "src_node", "dst_node", "confidence", "reason", "created_at", "updated_at")
] + [
    ("node", c)
    for c in ("node_id", "entity_uid", "canonical", "status", "promoted_at", "created_at")
] + [
    ("relation_resolution", "created_at"),
    ("relation_resolution", "updated_at"),
    ("schema_registry", "access_tier"),
    ("topic_registry", "created_at"),
    ("topic_alias", "created_at"),
]


def upgrade() -> None:
    run_sql_file("296_schema_comments_bak_cleanup.sql")


def downgrade() -> None:
    # 컬럼 COMMENT 원복(직전 NULL). 백업 테이블은 비가역 정리라 재생성하지 않는다.
    for table, col in _COMMENTED:
        op.execute(f"COMMENT ON COLUMN {table}.{col} IS NULL")
