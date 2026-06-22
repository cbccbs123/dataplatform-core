"""v290 — schema_registry access_tier 컬럼 + general/medical 시드 (spec 040)

Revision ID: v290_schema_registry_access_tier
Revises: v280_schema_registry_json_schema
"""
from __future__ import annotations

from alembic import op
from migrations.alembic._runsql import run_sql_file

revision = "v290_schema_registry_access_tier"
down_revision = "v280_schema_registry_json_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("290_schema_registry_access_tier.sql")


def downgrade() -> None:
    op.execute("ALTER TABLE schema_registry DROP COLUMN IF EXISTS access_tier")
