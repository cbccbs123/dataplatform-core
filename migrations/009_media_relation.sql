-- media_relation: 미디어 간 엣지 (카탈로그 relation_type_id 참조).
-- Requires: media_items, relation_type (004~008 적용 후).

CREATE TABLE IF NOT EXISTS media_relation (
    media_relation_id      BIGSERIAL PRIMARY KEY,
    source_media_item_id   BIGINT NOT NULL REFERENCES media_items (id) ON DELETE CASCADE,
    target_media_item_id   BIGINT NOT NULL REFERENCES media_items (id) ON DELETE CASCADE,
    relation_type_id       BIGINT NOT NULL REFERENCES relation_type (relation_type_id) ON DELETE RESTRICT,
    confidence             DOUBLE PRECISION,
    reason                   TEXT,
    status                   VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'superseded', 'rejected')),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at               TIMESTAMPTZ,
    CONSTRAINT chk_media_relation_no_self CHECK (source_media_item_id <> target_media_item_id),
    CONSTRAINT uq_media_relation_edge UNIQUE (source_media_item_id, target_media_item_id, relation_type_id)
);

CREATE INDEX IF NOT EXISTS idx_media_relation_source ON media_relation (source_media_item_id);
CREATE INDEX IF NOT EXISTS idx_media_relation_target ON media_relation (target_media_item_id);
CREATE INDEX IF NOT EXISTS idx_media_relation_type ON media_relation (relation_type_id);
