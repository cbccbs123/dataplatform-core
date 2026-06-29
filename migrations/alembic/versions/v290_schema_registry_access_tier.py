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
    # upgrade 가 INSERT 한 medical 시드를 먼저 지워야 v280 상태로 완전 복원된다(가역성).
    # medical 도메인은 v290 이전에 없었으므로 도메인 단위 DELETE 로 v290 INSERT 만 제거(v220 general 관례와 동일).
    op.execute("DELETE FROM schema_registry WHERE domain = 'medical'")
    op.execute("ALTER TABLE schema_registry DROP COLUMN IF EXISTS access_tier")
