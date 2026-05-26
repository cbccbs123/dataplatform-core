"""F-4.1 통합 스키마 — Medical ER 6 테이블 (DDL 전용)

Revision ID: v110_asset_medical_er
Revises: v100_asset_core
Create Date: 2026-05-22
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v110_asset_medical_er"
down_revision = "v100_asset_core"
branch_labels = None
depends_on = None

_TABLES = (
    "match_evidence",
    "match_decision",
    "match_candidate",
    "entity_edge",
    "asset_entity_link",
    "entity",
)


def upgrade() -> None:
    run_sql_file("110_asset_medical_er.sql")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
