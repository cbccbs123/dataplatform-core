"""F-4.1 통합 스키마 — Core 8 테이블

Revision ID: v100_asset_core
Revises:
Create Date: 2026-05-22
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v100_asset_core"
down_revision = None
branch_labels = None
depends_on = None

# downgrade 시 FK 의존성 역순으로 DROP.
_TABLES = (
    "asset_classification",
    "schema_registry",
    "access_log",
    "asset_lineage",
    "asset_embedding",
    "asset_metadata",
    "asset",
    "asset_group",
)


def upgrade() -> None:
    run_sql_file("100_asset_core.sql")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
