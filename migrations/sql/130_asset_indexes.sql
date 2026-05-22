-- =============================================================================
-- F-4.1 통합 스키마 — 인덱스 일괄 (BTree / GIN / BRIN / IVFFlat)
-- =============================================================================
-- 100~120 의 모든 테이블 생성 후 마지막에 적용.
-- =============================================================================

-- ---- BTree: FK·자주 필터되는 컬럼 -----------------------------------------
CREATE INDEX IF NOT EXISTS idx_asset_group_id     ON asset (group_id);
CREATE INDEX IF NOT EXISTS idx_asset_domain_label ON asset (domain_label);
CREATE INDEX IF NOT EXISTS idx_asset_modality     ON asset (modality);
CREATE INDEX IF NOT EXISTS idx_asset_status       ON asset (status);
CREATE INDEX IF NOT EXISTS idx_asset_file_hash    ON asset (file_hash);

CREATE INDEX IF NOT EXISTS idx_asset_classification_label ON asset_classification (final_label);

CREATE INDEX IF NOT EXISTS idx_entity_type ON entity (entity_type);

CREATE INDEX IF NOT EXISTS idx_asset_entity_link_asset  ON asset_entity_link (asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_entity_link_entity ON asset_entity_link (entity_id);

CREATE INDEX IF NOT EXISTS idx_entity_edge_source ON entity_edge (source_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_edge_target ON entity_edge (target_entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_edge_type   ON entity_edge (edge_type);

CREATE INDEX IF NOT EXISTS idx_match_candidate_left  ON match_candidate (left_asset_id);
CREATE INDEX IF NOT EXISTS idx_match_candidate_right ON match_candidate (right_asset_id);

CREATE INDEX IF NOT EXISTS idx_match_decision_left   ON match_decision (left_asset_id);
CREATE INDEX IF NOT EXISTS idx_match_decision_right  ON match_decision (right_asset_id);
CREATE INDEX IF NOT EXISTS idx_match_decision_policy ON match_decision (policy_version);

CREATE INDEX IF NOT EXISTS idx_match_evidence_decision ON match_evidence (decision_id);

CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue (status, priority_score DESC);

CREATE INDEX IF NOT EXISTS idx_content_cluster_channel ON content_cluster (channel, cluster_id);

-- ---- GIN: JSONB(jsonb_path_ops) · 배열 · tsvector ------------------------
CREATE INDEX IF NOT EXISTS idx_asset_metadata_core_meta
    ON asset_metadata USING GIN (core_meta jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_asset_metadata_ext_meta
    ON asset_metadata USING GIN (ext_meta jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_asset_metadata_tags
    ON asset_metadata USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_asset_metadata_search_vector
    ON asset_metadata USING GIN (search_vector);

-- ---- BRIN: append-only 시계열 컬럼 ---------------------------------------
CREATE INDEX IF NOT EXISTS idx_asset_created_at_brin
    ON asset USING BRIN (created_at);
CREATE INDEX IF NOT EXISTS idx_asset_lineage_occurred_at_brin
    ON asset_lineage USING BRIN (occurred_at);
CREATE INDEX IF NOT EXISTS idx_access_log_occurred_at_brin
    ON access_log USING BRIN (occurred_at);

-- ---- IVFFlat: 1536D 임베딩 코사인 유사도 (lists=200) ----------------------
CREATE INDEX IF NOT EXISTS idx_asset_embedding_vec_ivfflat
    ON asset_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);
