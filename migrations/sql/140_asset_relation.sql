-- =============================================================================
-- F-3.5 선행 — asset_relation (media_relation 의 asset FK 버전)
-- =============================================================================
-- 관계 카탈로그(relation_kind/relation_topic_parent/relation_subtopic/relation_type)는
-- 도메인 무관 공유 테이블이므로 그대로 재사용한다(별도 마이그레이션 004~009 로 이미 존재).
-- asset_relation 은 OLD media_relation(009) 구조를 source/target 을 asset(asset_id) 참조로 바꾼 것.
-- 적용 순서: 140 은 100(core, asset) 이후 + relation_type 카탈로그 존재 전제.
-- =============================================================================

CREATE TABLE IF NOT EXISTS asset_relation (
    relation_id      BIGSERIAL PRIMARY KEY,
    source_asset_id  BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    target_asset_id  BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    relation_type_id BIGINT NOT NULL REFERENCES relation_type (relation_type_id) ON DELETE RESTRICT,
    confidence       DOUBLE PRECISION,
    reason           TEXT,
    status           VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'rejected')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ,
    CONSTRAINT chk_asset_relation_no_self CHECK (source_asset_id <> target_asset_id),
    CONSTRAINT uq_asset_relation_edge UNIQUE (source_asset_id, target_asset_id, relation_type_id)
);

CREATE INDEX IF NOT EXISTS idx_asset_relation_source ON asset_relation (source_asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_relation_target ON asset_relation (target_asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_relation_type   ON asset_relation (relation_type_id);
