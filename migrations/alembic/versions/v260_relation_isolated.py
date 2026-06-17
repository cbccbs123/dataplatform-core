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
    # 비대칭(의도적): upgrade backfill 은 failed+'isolated:no_edges' 만 isolated 로 복구하지만, downgrade 는
    # 모든 isolated 를 failed 로 접는다(고립의 종전 표현이 pending/failed 양쪽이라 정확한 역매핑 불가).
    # 'isolated'를 먼저 흡수해야 3-어휘 CHECK 재적용 시 위반이 없다. 라운드트립 안전: 재 upgrade 시 backfill
    # 키('isolated:no_edges')가 다시 일치해 복구된다(reason 은 downgrade 가 건드리지 않음).
    op.execute("UPDATE relation_resolution SET status = 'failed' WHERE status = 'isolated'")
    op.execute(
        "ALTER TABLE relation_resolution "
        "DROP CONSTRAINT IF EXISTS relation_resolution_status_check"
    )
    op.execute(
        "ALTER TABLE relation_resolution ADD CONSTRAINT relation_resolution_status_check "
        "CHECK (status IN ('pending', 'resolved', 'failed'))"
    )
