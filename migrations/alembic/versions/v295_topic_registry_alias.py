"""v295 — topic_registry·topic_alias (spec 058 관계 topic 정규화·정본 레지스트리+alias 캐시)

Revision ID: v295_topic_registry_alias
Revises: v294_graph_edge_topic_index

관계 생성의 자유 기입 topic 을 성장하는 정본 레지스트리로 수렴시키는 2데이터 모델.
DDL 본문은 migrations/sql/295_topic_registry_alias.sql 단일 출처(멱등·IF NOT EXISTS·
pgvector cosine 인덱스). downgrade 는 두 테이블 drop(가역) — alias 가 registry 를 FK 참조하므로
자식(topic_alias) 먼저, 부모(topic_registry) 나중에 드롭한다.

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v295_topic_registry_alias"
down_revision = "v294_graph_edge_topic_index"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("295_topic_registry_alias.sql")


def downgrade() -> None:
    # FK 의존 순서: 자식(alias) → 부모(registry). 인덱스는 테이블 drop 시 함께 제거됨.
    op.execute("DROP TABLE IF EXISTS topic_alias")
    op.execute("DROP TABLE IF EXISTS topic_registry")
