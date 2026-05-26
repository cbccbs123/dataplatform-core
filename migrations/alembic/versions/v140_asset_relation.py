"""F-3.5 선행 — asset_relation (media_relation 의 asset FK 버전)

Revision ID: v140_asset_relation
Revises: v130_asset_indexes
Create Date: 2026-05-22

전제: relation_type 카탈로그(migrations/004~009)는 이미 존재해야 한다(도메인 무관 공유 테이블).
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v140_asset_relation"
down_revision = "v130_asset_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("140_asset_relation.sql")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS asset_relation CASCADE")
