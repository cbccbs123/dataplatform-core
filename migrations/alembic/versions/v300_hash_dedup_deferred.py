"""v300 — 해시 dedup 유니크 인덱스 deferred 확장 (069 US-B FR-B11 · P2-8)

Revision ID: v300_hash_dedup_deferred
Revises: v299_asset_topic

기존 부분 유니크 인덱스(150 · registered 만)를 registered+deferred 로 재정의한다 — 앱 dedup
사전조회(009: registered+deferred)와 DB 안전망의 범위 불일치로, 병렬 수집이 동시에 통과하면
deferred 중복 행이 영속될 수 있던 틈을 닫는다. DDL 본문은 migrations/sql/300_hash_dedup_deferred.sql
단일 출처(run_sql_file 관례). 선행 확인(2026-07-15 dev): registered/deferred 중복 해시 0건.

downgrade 는 신규 인덱스를 지우고 150 원형(registered 만)을 복원 — 가역(데이터 무접촉·인덱스만).

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v300_hash_dedup_deferred"
down_revision = "v299_asset_topic"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("300_hash_dedup_deferred.sql")


def downgrade() -> None:
    # 신규 인덱스 제거 후 150 원형(registered 전용) 복원 — 인덱스만 건드리는 가역 롤백.
    op.execute("DROP INDEX IF EXISTS uq_asset_file_hash_dedup")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_asset_file_hash_registered "
        "ON asset (file_hash) WHERE status = 'registered' AND file_hash IS NOT NULL"
    )
    # migration-reviewer 권고: 150 원형의 COMMENT 까지 복원해 완전 원형 롤백(기능 무관·문서 정합).
    op.execute(
        "COMMENT ON INDEX uq_asset_file_hash_registered IS "
        "'registered 자산은 동일 file_hash 중복 불가(내용 기반 dedup 안전망).'"
    )
