"""v299 — asset_topic 자산 자기주제 정본 테이블 (spec 065 · FR-101/102)

Revision ID: v299_asset_topic
Revises: v298_labels_schema_object

자산 자기 내용(summary/keywords/labels) 하이브리드 분류(임베딩 kNN → LLM 닫힌 확정 → 058 canonicalize)의
(topic, subtopic) 정본을 담는 신규 테이블을 추가한다. 검색·포탈이 쓰던 관계-이웃 투영(project_asset_topics)을
대체할 단일 주제 소스다(소비 스왑은 후속 G3). DDL 본문은 migrations/sql/299_asset_topic.sql 단일 출처
(run_sql_file 관례). 신규 테이블·인덱스만 추가하고 다른 스키마는 무접촉이다.

downgrade 는 asset_topic 테이블을 DROP(인덱스는 테이블과 함께 소멸) — 가역. 데이터가 있어도 신규 테이블만
제거하므로 다른 테이블·이력에 영향 없다.

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v299_asset_topic"
down_revision = "v298_labels_schema_object"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("299_asset_topic.sql")


def downgrade() -> None:
    # 신규 테이블만 제거(가역) — 인덱스는 테이블과 함께 소멸.
    op.execute("DROP TABLE IF EXISTS asset_topic")
