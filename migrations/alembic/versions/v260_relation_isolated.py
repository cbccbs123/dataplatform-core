"""v260 — relation_resolution.status 에 'isolated' 추가 (spec 035 #2 고립≠실패)

Revision ID: v260_relation_isolated
Revises: v250_relation_resolution
Create Date: 2026-06-17

고립(후보0/엣지0·예외 없음)을 일시 실패(예외)와 분리한다. status CHECK 어휘에 'isolated' 추가
+ DLQ(failed)에 잘못 격리된 고립 행(last_reason='isolated:no_edges') 복구(멱등). 부분 인덱스·스캔·
asset 계층 무손상. DDL 정본: migrations/sql/260_relation_resolution_isolated_status.sql.
"""
from __future__ import annotations

from alembic import op
from migrations.alembic._runsql import run_sql_file

revision = "v260_relation_isolated"
down_revision = "v250_relation_resolution"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("260_relation_resolution_isolated_status.sql")


def downgrade() -> None:
    # 'isolated' 행을 되돌릴 곳이 모호하므로(고립의 종전 표현은 pending/failed 양쪽) 안전하게 'failed'로
    # 접고 CHECK 를 원복한다. 'isolated'를 'failed'로 흡수해야 신규 CHECK 제거 시 위반이 없다.
    op.execute("UPDATE relation_resolution SET status = 'failed' WHERE status = 'isolated'")
    op.execute("ALTER TABLE relation_resolution DROP CONSTRAINT IF EXISTS relation_resolution_status_check")
    op.execute(
        "ALTER TABLE relation_resolution ADD CONSTRAINT relation_resolution_status_check "
        "CHECK (status IN ('pending', 'resolved', 'failed'))"
    )
