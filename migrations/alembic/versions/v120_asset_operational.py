"""F-4.1 통합 스키마 — 운영 3 + content_cluster 1 테이블

Revision ID: v120_asset_operational
Revises: v110_asset_medical_er
Create Date: 2026-05-22
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v120_asset_operational"
down_revision = "v110_asset_medical_er"
branch_labels = None
depends_on = None

_TABLES = (
    "content_cluster",
    "er_policy",
    "unresolved_pool",
    "review_queue",
)


def upgrade() -> None:
    run_sql_file("120_asset_operational.sql")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
