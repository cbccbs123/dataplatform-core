"""v294 — graph_edge.topic 표현식 인덱스 2 (spec 056·주제 검색·탐색 seam 가속)

Revision ID: v294_graph_edge_topic_index
Revises: v293_lineage_activity_idx

topic_query(find_topic_neighbors·list_topics·assets_in_topic) 의 topic->>'topic_ko' /
topic->>'subtopic_ko' 등가 술어를 인덱스 프로브화(SC-06 성능). DDL 본문은
migrations/sql/294_graph_edge_topic_index.sql 단일 출처(멱등·IF NOT EXISTS).

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op
from migrations.alembic._runsql import run_sql_file

revision = "v294_graph_edge_topic_index"
down_revision = "v293_lineage_activity_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("294_graph_edge_topic_index.sql")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_graph_edge_topic_ko")
    op.execute("DROP INDEX IF EXISTS ix_graph_edge_subtopic_ko")
