"""F-4.1 통합 스키마 — 인덱스 일괄 (BTree / GIN / BRIN / IVFFlat)

Revision ID: v130_asset_indexes
Revises: v120_asset_operational
Create Date: 2026-05-22
"""
from __future__ import annotations

from alembic import op

from migrations.alembic._runsql import run_sql_file

revision = "v130_asset_indexes"
down_revision = "v120_asset_operational"
branch_labels = None
depends_on = None

# downgrade 시 제거할 인덱스(테이블 DROP 으로 함께 사라지지만, 인덱스만 되돌릴 때를 위해 명시).
_INDEXES = (
    "idx_asset_embedding_vec_ivfflat",
    "idx_access_log_occurred_at_brin",
    "idx_asset_lineage_occurred_at_brin",
    "idx_asset_created_at_brin",
    "idx_asset_metadata_search_vector",
    "idx_asset_metadata_tags",
    "idx_asset_metadata_ext_meta",
    "idx_asset_metadata_core_meta",
    "idx_content_cluster_channel",
    "idx_review_queue_status",
    "idx_match_evidence_decision",
    "idx_match_decision_policy",
    "idx_match_decision_right",
    "idx_match_decision_left",
    "idx_match_candidate_right",
    "idx_match_candidate_left",
    "idx_entity_edge_type",
    "idx_entity_edge_target",
    "idx_entity_edge_source",
    "idx_asset_entity_link_entity",
    "idx_asset_entity_link_asset",
    "idx_entity_type",
    "idx_asset_classification_label",
    "idx_asset_file_hash",
    "idx_asset_status",
    "idx_asset_modality",
    "idx_asset_domain_label",
    "idx_asset_group_id",
)


def upgrade() -> None:
    run_sql_file("130_asset_indexes.sql")


def downgrade() -> None:
    for index in _INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {index}")
