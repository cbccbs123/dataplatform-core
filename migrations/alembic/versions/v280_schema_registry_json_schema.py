"""v280 — general ext_meta json_schema 값 검증용 완성 (spec 039)

Revision ID: v280_schema_registry_json_schema
Revises: v270_drop_search_vector
"""
from __future__ import annotations

from alembic import op
from migrations.alembic._runsql import run_sql_file

revision = "v280_schema_registry_json_schema"
down_revision = "v270_drop_search_vector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("280_schema_registry_json_schema.sql")


def downgrade() -> None:
    op.execute(
        "UPDATE schema_registry SET json_schema = '{\"type\":\"array\"}'::jsonb "
        "WHERE domain = 'general' AND meta_key IN ('labels','objects','keyframes')"
    )
