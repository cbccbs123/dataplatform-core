"""C+ — graph_edge relation_kind 직접 참조 + topic jsonb, 카탈로그 3테이블 드롭

Revision ID: v230_graph_edge_kind_topic
Revises: v220_schema_registry_seed
Create Date: 2026-05-28
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v230_graph_edge_kind_topic"
down_revision = "v220_schema_registry_seed"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("230_graph_edge_kind_topic.sql")


def downgrade() -> None:
    # 구조 복원(데이터 무손실 보장 안 함 — topic jsonb→정규화 역변환은 손실).
    op.execute("DROP INDEX IF EXISTS idx_graph_edge_kind")
    op.execute("ALTER TABLE graph_edge DROP CONSTRAINT IF EXISTS uq_graph_edge_kind")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_topic_parent (
            topic_id UUID PRIMARY KEY,
            topic_ko VARCHAR(200) NOT NULL DEFAULT '',
            topic_en VARCHAR(200) NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_relation_topic_parent_ko_en UNIQUE (topic_ko, topic_en))
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_subtopic (
            subtopic_id UUID PRIMARY KEY,
            topic_id UUID NOT NULL REFERENCES relation_topic_parent (topic_id) ON DELETE RESTRICT,
            subtopic_ko VARCHAR(200) NOT NULL DEFAULT '',
            subtopic_en VARCHAR(200) NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_relation_subtopic_topic_sub_ko_en UNIQUE (topic_id, subtopic_ko, subtopic_en))
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS relation_type (
            relation_type_id UUID PRIMARY KEY,
            relation_kind_id UUID NOT NULL REFERENCES relation_kind (relation_kind_id) ON DELETE RESTRICT,
            relation_subtopic_id UUID NOT NULL REFERENCES relation_subtopic (subtopic_id) ON DELETE RESTRICT,
            status VARCHAR(20) NOT NULL DEFAULT 'inactive' CHECK (status IN ('active','inactive')),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT uq_relation_type_kind_subtopic UNIQUE (relation_kind_id, relation_subtopic_id))
        """
    )
    op.execute("ALTER TABLE graph_edge ADD COLUMN IF NOT EXISTS relation_type_id UUID")
    op.execute("ALTER TABLE graph_edge DROP COLUMN IF EXISTS topic")
    op.execute("ALTER TABLE graph_edge DROP COLUMN IF EXISTS relation_kind_id")
    # HITL status 되돌림(구조만)
    op.execute("ALTER TABLE graph_edge DROP CONSTRAINT IF EXISTS graph_edge_status_check")
    op.execute("ALTER TABLE graph_edge ALTER COLUMN status SET DEFAULT 'active'")
    op.execute("ALTER TABLE graph_edge ADD CONSTRAINT graph_edge_status_check "
               "CHECK (status IN ('active','superseded','rejected'))")
    op.execute("ALTER TABLE graph_edge DROP COLUMN IF EXISTS reviewed_by")
    op.execute("ALTER TABLE graph_edge DROP COLUMN IF EXISTS reviewed_at")
