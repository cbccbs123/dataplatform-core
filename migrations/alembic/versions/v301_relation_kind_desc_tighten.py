"""v301 — 명시적 3종 설명문 좁히기 (081 · 2026-08-03 측정 채택)

Revision ID: v301_relation_kind_desc
Revises: v300_hash_dedup_deferred

`relation_kind.description` 은 **LLM 프롬프트에 그대로 실린다**(관계 카탈로그 블록). 명시적 3종
(same_series·references·derived_from)의 설명에 있던 "라인업·묶음" 이라는 느슨한 표현이 LLM 에게
*"같은 범주에 속하는 것들"* 로 읽혀 오분류를 만들었다(창덕궁↔덕수궁을 "연작" 등 · 3종 정확도
25~33%). 종류 정의가 프롬프트 세 곳에 흩어져 어긋난 것이 원인이고, 그중 옳은 정의(엣지케이스
안내의 "같은 stem + 순번/버전")에 코드 힌트와 이 컬럼을 맞춘다. DDL 본문은
migrations/sql/301_relation_kind_desc_tighten.sql 단일 출처(run_sql_file 관례).

**데이터 무접촉** — description 텍스트만 갱신한다(kind_id·status·graph_edge 무변경). 스키마 변경도
없다. 즉 이 리비전은 "프롬프트 입력 문구"를 바꾸는 것이고, 이미 만들어진 관계에는 영향이 없다
(새 문구는 다음 관계 생성부터 적용).

downgrade 는 이전 문구를 그대로 복원 — 가역. 원문구는 seed(`scripts/seed_topic_registry` 계열이
아니라 relation_kind 초기 시드)와 이 파일에만 남으므로 여기 문자열이 복원 정본이다.

주의: revision ID 는 alembic_version.version_num(VARCHAR(32)) 에 저장되므로 32자 이하로 유지한다.
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v301_relation_kind_desc"
down_revision = "v300_hash_dedup_deferred"
branch_labels = None
depends_on = None

# 좁히기 **이전** 문구 — downgrade 복원 정본(이 값이 곧 원형이다).
_PREV: dict[str, str] = {
    "same_series": "같은 시리즈·연작·라인업 연결",
    "references": "명시적 인용·링크·제목 참조",
    "derived_from": "한 콘텐츠가 다른 콘텐츠에서 파생",
}


def upgrade() -> None:
    run_sql_file("301_relation_kind_desc_tighten.sql")


def downgrade() -> None:
    # 문구만 되돌린다(데이터 무접촉·멱등). 파라미터 바인딩으로 인용 이스케이프를 피한다.
    for code, desc in _PREV.items():
        op.execute(
            f"UPDATE relation_kind SET description = '{desc}' WHERE kind_code = '{code}'"  # noqa: S608
        )
