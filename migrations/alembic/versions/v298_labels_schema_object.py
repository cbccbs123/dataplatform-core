"""v298 — labels ext_meta json_schema 교정(array of string → array of object) (spec 039 시드 버그)

Revision ID: v298_labels_schema_object
Revises: v297_topic_parent_scope

v280 이 labels items 를 {"type":"string"} 로 잘못 지정했으나 실제 데이터(image/video CLIP 제로샷)는
[{label, score}] 객체 배열이다. 039 값 검증 활성 신규 적재에서 image/video 가 전부 실패 → labels 스키마를
object{label:string, score:number}(label 필수)로 교정. DDL 본문은 migrations/sql/298_labels_schema_object.sql
단일 출처(멱등 UPDATE·schema_registry+ext_meta_field_registry·general/medical).

downgrade 는 v280 상태(array of string)로 원복(가역). 단 이미 객체 라벨이 적재된 뒤라면 그 데이터는
복원된 string 스키마와 비호환이나, downgrade 는 스키마 값만 되돌릴 뿐 데이터를 건드리지 않는다.

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v298_labels_schema_object"
down_revision = "v297_topic_parent_scope"
branch_labels = None
depends_on = None


def upgrade() -> None:
    run_sql_file("298_labels_schema_object.sql")


def downgrade() -> None:
    # v280 상태(array of string)로 원복 — 스키마 값만 되돌린다(데이터 무변경·가역).
    conn = op.get_bind()
    conn.exec_driver_sql(
        "UPDATE ext_meta_field_registry SET json_schema = "
        "'{\"type\":\"array\",\"items\":{\"type\":\"string\"}}'::jsonb "
        "WHERE domain IN ('general', 'medical') AND meta_key = 'labels'"
    )
    conn.exec_driver_sql(
        "UPDATE schema_registry SET json_schema = "
        "'{\"type\":\"array\",\"items\":{\"type\":\"string\"}}'::jsonb "
        "WHERE domain IN ('general', 'medical') AND meta_key = 'labels'"
    )
