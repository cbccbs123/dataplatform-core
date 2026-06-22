"""v291 — ext_meta_field_registry 신설 + schema_registry backfill (spec 041)

Revision ID: v291_ext_meta_field_registry
Revises: v290_schema_registry_access_tier
"""
from __future__ import annotations

from alembic import op
from migrations.alembic._runsql import run_sql_file

revision = "v291_ext_meta_field_registry"
down_revision = "v290_schema_registry_access_tier"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("291_ext_meta_field_registry.sql")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ext_meta_field_registry")
