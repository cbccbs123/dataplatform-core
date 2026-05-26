"""F-5.1 보강 — asset.status 'deferred' 추가

Revision ID: v160_asset_status_deferred
Revises: v150_asset_file_hash_dedup
Create Date: 2026-05-26
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v160_asset_status_deferred"
down_revision = "v150_asset_file_hash_dedup"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("160_asset_status_deferred.sql")


def downgrade() -> None:
    op.execute("ALTER TABLE asset DROP CONSTRAINT IF EXISTS asset_status_check")
    op.execute(
        "ALTER TABLE asset ADD CONSTRAINT asset_status_check "
        "CHECK (status IN ('received','routing','classifying','extracting','registered','failed'))"
    )
