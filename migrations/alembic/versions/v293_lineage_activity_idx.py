"""v293 — asset_lineage(activity, asset_id) 인덱스 (spec 054·스냅샷 버킷 EXISTS 가속)

Revision ID: v293_lineage_activity_idx
Revises: v292_modality_canonical

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v293_lineage_activity_idx"
down_revision = "v292_modality_canonical"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("293_lineage_activity_idx.sql")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_asset_lineage_activity_asset")
