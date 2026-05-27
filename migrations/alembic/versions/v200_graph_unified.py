"""v2 단계 C — 공용 그래프 계층(node + graph_edge), asset_relation 흡수

Revision ID: v200_graph_unified
Revises: v170_relation_catalog_seed
Create Date: 2026-05-27
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v200_graph_unified"
down_revision = "v170_relation_catalog_seed"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("200_graph_unified.sql")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS graph_edge CASCADE")
    op.execute("DROP TABLE IF EXISTS node CASCADE")
