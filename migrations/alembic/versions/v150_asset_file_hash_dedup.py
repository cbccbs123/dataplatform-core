"""F-1.3 보강 — 자산 file_hash 중복 방지 부분 유니크 인덱스

Revision ID: v150_asset_file_hash_dedup
Revises: v140_asset_relation
Create Date: 2026-05-26
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v150_asset_file_hash_dedup"
down_revision = "v140_asset_relation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("150_asset_file_hash_dedup.sql")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_asset_file_hash_registered")
