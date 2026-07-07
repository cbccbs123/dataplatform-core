"""v297 — topic_registry·topic_alias parent_topic 스코프(spec 058 v2 subtopic 부모 스코프)

Revision ID: v297_topic_parent_scope
Revises: v296_schema_comments

topic 층(닫힌 27+기타·parent_topic IS NULL)과 subtopic 층(열린 성장·부모 topic 스코프)을
분리하는 2층 구조. DDL 본문은 migrations/sql/297_topic_parent_scope.sql 단일 출처
(멱등·부분 유니크 인덱스 2개·FK 완화·상세 근거는 SQL 헤더 주석 참조).

downgrade 는 대칭 원복(가역): 부분 유니크 인덱스 드롭 → parent_topic 컬럼 드롭 →
원 제약(registry.topic_ko UNIQUE·alias.raw_ko PK·canonical_ko FK) 복원.
※ subtopic 층 행이 존재하는 상태(G10 이후)에서의 downgrade 는 topic_ko/raw_ko 중복으로
  원 유니크 제약 복원이 실패할 수 있다(정상 — 하위 데이터가 옛 스키마와 비호환). G9 적재 직후
  (registry 28 topic 층·alias 0)에서는 완전 가역이다.

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v297_topic_parent_scope"
down_revision = "v296_schema_comments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("297_topic_parent_scope.sql")


def downgrade() -> None:
    # 사전 가드(migration-reviewer 🟡): subtopic 층 행(parent_topic IS NOT NULL)이 있으면 2층 키공간을
    #   1층으로 무손실 투영할 수 없어 원 유니크/PK 복원이 topic_ko/raw_ko 중복으로 실패한다. 모호한
    #   "duplicate key" 대신 사유를 명확히 알리고 중단한다(수동 붕괴 정책 필요). 전부 NULL이면 통과.
    conn = op.get_bind()
    for tbl in ("topic_registry", "topic_alias"):
        n = conn.exec_driver_sql(
            f"SELECT count(*) FROM {tbl} WHERE parent_topic IS NOT NULL"
        ).scalar()
        if n:
            raise RuntimeError(
                f"v297 downgrade 불가: {tbl}에 subtopic 층 행 {n}건(parent_topic NOT NULL) 존재. "
                "2층→1층 무손실 투영 불가(topic_ko/raw_ko 중복). 하위 데이터를 먼저 정리·이관한 뒤 "
                "downgrade 하라(가역은 parent_topic 전부 NULL 상태에서만 성립)."
            )
    # 부분 유니크 인덱스 드롭(부모 스코프)
    op.execute("DROP INDEX IF EXISTS uq_topic_alias_parent_raw")
    op.execute("DROP INDEX IF EXISTS uq_topic_alias_raw_ko_root")
    op.execute("DROP INDEX IF EXISTS uq_topic_registry_parent_sub")
    op.execute("DROP INDEX IF EXISTS uq_topic_registry_topic_ko_root")
    # parent_topic 컬럼 드롭(원복) — 컬럼에 매인 잔여 인덱스도 함께 제거된다.
    op.execute("ALTER TABLE topic_alias    DROP COLUMN IF EXISTS parent_topic")
    op.execute("ALTER TABLE topic_registry DROP COLUMN IF EXISTS parent_topic")
    # 원 제약 복원: registry.topic_ko UNIQUE · alias.raw_ko PK · canonical_ko FK(가역).
    op.execute(
        "ALTER TABLE topic_registry ADD CONSTRAINT topic_registry_topic_ko_key UNIQUE (topic_ko)"
    )
    op.execute("ALTER TABLE topic_alias ADD CONSTRAINT topic_alias_pkey PRIMARY KEY (raw_ko)")
    op.execute(
        "ALTER TABLE topic_alias ADD CONSTRAINT topic_alias_canonical_ko_fkey "
        "FOREIGN KEY (canonical_ko) REFERENCES topic_registry (topic_ko)"
    )
