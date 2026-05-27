"""F-4.13 — general 도메인 ext_meta 허용 키 시드(schema_registry)

Revision ID: v220_schema_registry_seed
Revises: v210_drop_legacy
Create Date: 2026-05-27
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v220_schema_registry_seed"
down_revision = "v210_drop_legacy"
branch_labels = None
depends_on = None

_KEYS = ("summary", "keywords", "labels", "objects", "keyframes", "stt", "caption")


def upgrade() -> None:
    run_sql_file("220_schema_registry_seed.sql")


def downgrade() -> None:
    op.execute(
        "DELETE FROM schema_registry WHERE domain = 'general' AND meta_key IN "
        "('summary','keywords','labels','objects','keyframes','stt','caption')"
    )
