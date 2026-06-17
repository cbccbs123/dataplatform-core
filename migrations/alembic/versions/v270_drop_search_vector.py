"""v270 — asset_metadata.search_vector 제거 (spec 037 OpenSearch 전용·PG FTS 폐기)

Revision ID: v270_drop_search_vector
Revises: v260_relation_isolated
Create Date: 2026-06-17

검색을 OpenSearch 전용으로 전환(037)하면서 PG FTS(search_vector)를 읽던 경로(media_search)가
삭제됐다. 적재도 to_tsvector 를 더 이상 채우지 않으므로 search_vector 컬럼과 GIN 인덱스가 dead.
GIN 인덱스 + 컬럼을 드롭(멱등). downgrade 는 구조(컬럼·인덱스)만 복원(데이터 비복원 — 재적재 필요).
DDL 정본: migrations/sql/270_drop_search_vector.sql.
"""
from __future__ import annotations

from alembic import op
from migrations.alembic._runsql import run_sql_file

revision = "v270_drop_search_vector"
down_revision = "v260_relation_isolated"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("270_drop_search_vector.sql")


def downgrade() -> None:
    # 구조만 복원(데이터 비복원): 컬럼 재생성 후 GIN 인덱스 재생성. 기존 행의 search_vector 는
    # NULL 로 남으므로 FTS 를 다시 쓰려면 to_tsvector 재적재(또는 백필)가 필요하다.
    op.execute("ALTER TABLE asset_metadata ADD COLUMN search_vector TSVECTOR")
    op.execute(
        "CREATE INDEX idx_asset_metadata_search_vector "
        "ON asset_metadata USING GIN (search_vector)"
    )
