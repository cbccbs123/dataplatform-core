"""v292 — asset.modality canonical 4종 정규화 (spec 053)

Revision ID: v292_modality_canonical
Revises: v291_ext_meta_field_registry
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v292_modality_canonical"
down_revision = "v291_ext_meta_field_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # DROP → UPDATE(txt/pdf/json/word/excel/powerpoint→text) → ADD CHECK → COMMENT → ANALYZE
    run_sql_file("292_modality_canonical.sql")


def downgrade() -> None:
    # 순서: canonical CHECK 제거 → 'text'를 file_kind 로 best-effort 역매핑 → 10종 CHECK 복원.
    # 'text'→원 file_kind 완전복원은 손실적(txt vs json 구분 불가) → fs_path 확장자 기반 역매핑.
    # 확장자가 없거나 매핑 밖(.md/.csv 등)인 텍스트 자산은 모두 'txt'로 강제 귀결(무해·복원 CHECK 통과).
    op.execute("ALTER TABLE asset DROP CONSTRAINT IF EXISTS asset_modality_check")
    op.execute(
        """
        UPDATE asset SET modality = CASE
            WHEN modality <> 'text' THEN modality
            WHEN lower(fs_path) LIKE '%.pdf'  THEN 'pdf'
            WHEN lower(fs_path) LIKE '%.json' THEN 'json'
            WHEN lower(fs_path) LIKE '%.docx' OR lower(fs_path) LIKE '%.doc' THEN 'word'
            WHEN lower(fs_path) LIKE '%.xlsx' OR lower(fs_path) LIKE '%.xls' THEN 'excel'
            WHEN lower(fs_path) LIKE '%.pptx' OR lower(fs_path) LIKE '%.ppt' THEN 'powerpoint'
            ELSE 'txt'
        END
        WHERE modality = 'text'
        """
    )
    op.execute(
        """
        ALTER TABLE asset ADD CONSTRAINT asset_modality_check
            CHECK (modality IN ('txt', 'pdf', 'json', 'word', 'excel', 'powerpoint',
                                 'image', 'video', 'audio', 'unknown'))
        """
    )
    op.execute(
        "COMMENT ON COLUMN asset.modality IS "
        "'모달리티(MediaKind/OfficeKind 값): txt/pdf/json/word/excel/powerpoint/image/video/audio/unknown.'"
    )
