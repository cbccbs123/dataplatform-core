"""F-3.5 — 관계 카탈로그 기본 시드 (5종 kind + '일반' 토픽 + active relation_type)

Revision ID: v170_relation_catalog_seed
Revises: v160_asset_status_deferred
Create Date: 2026-05-26
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v170_relation_catalog_seed"
down_revision = "v160_asset_status_deferred"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("170_relation_catalog_seed.sql")


def downgrade() -> None:
    # 시드 카탈로그 제거(엣지가 참조 중이면 FK 로 막힘 — 의도된 안전장치).
    op.execute(
        "DELETE FROM relation_type rt USING relation_kind k "
        "WHERE rt.relation_kind_id = k.relation_kind_id "
        "AND k.kind_code IN ('same_domain','same_series','duplicate_near','references','derived_from')"
    )
    op.execute(
        "DELETE FROM relation_kind "
        "WHERE kind_code IN ('same_domain','same_series','duplicate_near','references','derived_from')"
    )
    op.execute(
        "DELETE FROM relation_subtopic s USING relation_topic_parent p "
        "WHERE s.topic_id = p.topic_id AND p.topic_ko = '일반' AND p.topic_en = 'general'"
    )
    op.execute("DELETE FROM relation_topic_parent WHERE topic_ko = '일반' AND topic_en = 'general'")
