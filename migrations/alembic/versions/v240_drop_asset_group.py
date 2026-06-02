"""v240 — 자산 그룹화 철회: asset_group / asset.group_id / idx_asset_group_id 제거

Revision ID: v240_drop_asset_group
Revises: v230_graph_edge_kind_topic
Create Date: 2026-06-02

경로 기반 자동 그룹화(구 spec 007) 철회. 다운로드 묶음은 관계(graph_edge) 기반
ego-network 로 단일화(spec 010 US3·FR-008). 의료 환자·검사 묶음은 단계 D 엔티티 ER 담당.
근거 ADR: docs/decisions/2026-06-02-relation-based-download-bundles.md
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v240_drop_asset_group"
down_revision = "v230_graph_edge_kind_topic"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("240_drop_asset_group.sql")


def downgrade() -> None:
    # 구조 복원(v100/v130 정의 동형, 가역). 데이터 무손실은 보장하지 않는다 —
    # 제거된 group 배정(asset_group 행·asset.group_id 값)은 복구되지 않는다.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS asset_group (
            group_id   UUID PRIMARY KEY,
            group_key  VARCHAR(255) NOT NULL,
            group_kind VARCHAR(50) NOT NULL DEFAULT 'general'
                CHECK (group_kind IN ('general', 'study_uid', 'mrn', 'manual')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_asset_group_kind_key UNIQUE (group_kind, group_key))
        """
    )
    op.execute(
        "ALTER TABLE asset ADD COLUMN IF NOT EXISTS group_id "
        "UUID REFERENCES asset_group (group_id) ON DELETE SET NULL"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_asset_group_id ON asset (group_id)")
