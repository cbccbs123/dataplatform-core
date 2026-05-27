"""v2 단계 C 정리 — 일반 도메인 미사용(레거시/의료/ER/옵션B) 테이블 드랍

Revision ID: v210_drop_legacy
Revises: v200_graph_unified
Create Date: 2026-05-27

의료(단계 D)·옵션 B 착수 시 해당 단계 마이그레이션에서 entity/match_*/er_policy 등을 재도입한다.
forward-only 정리이므로 downgrade 는 no-op.
"""
from __future__ import annotations

from migrations.alembic._runsql import run_sql_file

revision = "v210_drop_legacy"
down_revision = "v200_graph_unified"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("210_drop_legacy_tables.sql")


def downgrade() -> None:
    # forward-only 정리. 재도입은 단계 D/옵션 B 마이그레이션에서.
    pass
