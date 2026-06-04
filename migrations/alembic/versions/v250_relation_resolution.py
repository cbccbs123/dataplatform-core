"""v250 — 관계 재시도/미해소 큐 relation_resolution 신설 (spec 009 US3)

Revision ID: v250_relation_resolution
Revises: v240_drop_asset_group
Create Date: 2026-06-04

cross-asset 관계 생성(run_relations)의 미해소(일시 실패·고립) 자산을 추적하는 경량 큐.
status pending/resolved/failed(DLQ) + WHERE status='pending' 부분 인덱스. asset_id 자연키 PK,
FK→asset ON DELETE CASCADE. asset.status FSM·CHECK 제약은 무손상(관계 생성은 registered 이후 단계).
DDL 정본: migrations/sql/250_relation_resolution.sql.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v250_relation_resolution"
down_revision = "v240_drop_asset_group"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("250_relation_resolution.sql")


def downgrade() -> None:
    # 테이블 DROP 시 부분 인덱스도 자동 제거되지만, 가독성을 위해 명시적으로 먼저 드롭한다.
    # 이 큐는 asset 의 파생 메타라 드롭해도 자산 계층·그래프 계층에 영향이 없다(가역, 데이터만 손실).
    op.execute("DROP INDEX IF EXISTS idx_relation_resolution_pending")
    op.execute("DROP TABLE IF EXISTS relation_resolution")
