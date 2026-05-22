-- =============================================================================
-- F-3.5 선행 — asset_relation (media_relation 의 asset FK 버전)
-- =============================================================================
-- 관계 카탈로그(relation_kind/relation_topic_parent/relation_subtopic/relation_type)는
-- 도메인 무관 공유 테이블이므로 그대로 재사용한다(별도 마이그레이션 004~009 로 이미 존재).
-- asset_relation 은 OLD media_relation(009) 구조를 source/target 을 asset(asset_id) 참조로 바꾼 것.
-- 적용 순서: 140 은 100(core, asset) 이후.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 신규 DB 부트스트랩 가드: 관계 카탈로그 4테이블(레거시 004~009 가 권위 소스).
-- 레거시 SQL 을 적용한 DB 에서는 IF NOT EXISTS 로 no-op 이며, alembic 만으로
-- 부트스트랩하는 빈 DB 에서도 asset_relation 의 FK 가 성립하도록 테이블만 보장한다.
-- (seed 행은 권위 소스인 레거시 004 가 담당; 여기서는 시드하지 않는다.)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS relation_kind (
    relation_kind_id BIGSERIAL PRIMARY KEY,
    kind_code        VARCHAR(100) NOT NULL UNIQUE,
    kind_name_ko     VARCHAR(255) NOT NULL,
    description      TEXT,
    is_symmetric     BOOLEAN NOT NULL DEFAULT false,
    status           VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'inactive')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS relation_topic_parent (
    topic_id   BIGSERIAL PRIMARY KEY,
    topic_ko   VARCHAR(200) NOT NULL DEFAULT '',
    topic_en   VARCHAR(200) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_relation_topic_parent_ko_en UNIQUE (topic_ko, topic_en)
);

CREATE TABLE IF NOT EXISTS relation_subtopic (
    subtopic_id BIGSERIAL PRIMARY KEY,
    topic_id    BIGINT NOT NULL REFERENCES relation_topic_parent (topic_id) ON DELETE RESTRICT,
    subtopic_ko VARCHAR(200) NOT NULL DEFAULT '',
    subtopic_en VARCHAR(200) NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_relation_subtopic_topic_sub_ko_en UNIQUE (topic_id, subtopic_ko, subtopic_en)
);

CREATE TABLE IF NOT EXISTS relation_type (
    relation_type_id     BIGSERIAL PRIMARY KEY,
    relation_kind_id     BIGINT NOT NULL REFERENCES relation_kind (relation_kind_id) ON DELETE RESTRICT,
    relation_subtopic_id BIGINT NOT NULL REFERENCES relation_subtopic (subtopic_id) ON DELETE RESTRICT,
    status               VARCHAR(20) NOT NULL DEFAULT 'inactive'
        CHECK (status IN ('active', 'inactive')),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_relation_type_kind_subtopic UNIQUE (relation_kind_id, relation_subtopic_id)
);

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
