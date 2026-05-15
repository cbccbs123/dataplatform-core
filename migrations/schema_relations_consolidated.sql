-- =============================================================================
-- 관계 카탈로그 스키마 (통합 DDL, 최종 형태)
-- =============================================================================
-- media_items / media_chunks / relation_proposal / (구)relation_topic 제외. ``media_relation`` 은 009 이후 포함.
-- 내용은 migrations 004 + 005 + 006 + 007 + 008 + 009(미디어 엣지) 적용 후와 동등한 논리 스키마이다.
-- 용도: 문서·신규 DB 부트스트랩 참고. 이미 004~009를 적용한 DB에는 그대로 실행하지 말 것(중복).
-- =============================================================================

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

CREATE INDEX IF NOT EXISTS idx_relation_kind_status ON relation_kind (status);

CREATE TABLE IF NOT EXISTS relation_topic_parent (
    topic_id   BIGSERIAL PRIMARY KEY,
    topic_ko   VARCHAR(200) NOT NULL DEFAULT '',
    topic_en   VARCHAR(200) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_relation_topic_parent_ko_en UNIQUE (topic_ko, topic_en)
);

CREATE INDEX IF NOT EXISTS idx_relation_topic_parent_topic_ko
    ON relation_topic_parent (topic_ko);

CREATE TABLE IF NOT EXISTS relation_subtopic (
    subtopic_id BIGSERIAL PRIMARY KEY,
    topic_id    BIGINT NOT NULL REFERENCES relation_topic_parent (topic_id) ON DELETE RESTRICT,
    subtopic_ko VARCHAR(200) NOT NULL DEFAULT '',
    subtopic_en VARCHAR(200) NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_relation_subtopic_topic_sub_ko_en UNIQUE (topic_id, subtopic_ko, subtopic_en)
);

CREATE INDEX IF NOT EXISTS idx_relation_subtopic_topic_id ON relation_subtopic (topic_id);

CREATE TABLE IF NOT EXISTS relation_type (
    relation_type_id     BIGSERIAL PRIMARY KEY,
    relation_kind_id    BIGINT NOT NULL REFERENCES relation_kind (relation_kind_id) ON DELETE RESTRICT,
    relation_subtopic_id BIGINT NOT NULL REFERENCES relation_subtopic (subtopic_id) ON DELETE RESTRICT,
    status               VARCHAR(20) NOT NULL DEFAULT 'inactive'
        CHECK (status IN ('active', 'inactive')),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_relation_type_kind_subtopic UNIQUE (relation_kind_id, relation_subtopic_id)
);

CREATE INDEX IF NOT EXISTS idx_relation_type_status ON relation_type (status);
CREATE INDEX IF NOT EXISTS idx_relation_type_kind ON relation_type (relation_kind_id);
CREATE INDEX IF NOT EXISTS idx_relation_type_subtopic ON relation_type (relation_subtopic_id);

-- ---------------------------------------------------------------------------
-- Seeds (004와 동등)
-- ---------------------------------------------------------------------------
INSERT INTO relation_kind (kind_code, kind_name_ko, description, is_symmetric, status)
VALUES
    ('same_domain', '동일 도메인', '주제·분야가 같은 연결', true, 'active'),
    ('same_series', '동일 시리즈', '같은 시리즈·연작·라인업 연결', true, 'active'),
    ('duplicate_near', '유사 근접', '내용/주제/장면의 근접 유사', true, 'active'),
    ('references', '참조', '명시적 인용·링크·제목 참조', false, 'active'),
    ('derived_from', '파생', '한 콘텐츠가 다른 콘텐츠에서 파생', false, 'active')
ON CONFLICT (kind_code) DO NOTHING;

INSERT INTO relation_topic_parent (topic_ko, topic_en)
VALUES ('일반', 'general')
ON CONFLICT (topic_ko, topic_en) DO NOTHING;

INSERT INTO relation_subtopic (topic_id, subtopic_ko, subtopic_en)
SELECT p.topic_id, '', ''
FROM relation_topic_parent p
WHERE p.topic_ko = '일반' AND p.topic_en = 'general'
ON CONFLICT (topic_id, subtopic_ko, subtopic_en) DO NOTHING;

INSERT INTO relation_type (relation_kind_id, relation_subtopic_id, status)
SELECT k.relation_kind_id,
       s.subtopic_id,
       'active'
FROM relation_kind k
CROSS JOIN relation_subtopic s
JOIN relation_topic_parent p ON p.topic_id = s.topic_id
WHERE p.topic_ko = '일반'
  AND p.topic_en = 'general'
  AND s.subtopic_ko = ''
  AND s.subtopic_en = ''
  AND k.kind_code IN (
      'same_domain',
      'same_series',
      'duplicate_near',
      'references',
      'derived_from'
  )
  AND k.status = 'active'
ON CONFLICT (relation_kind_id, relation_subtopic_id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- media_relation (009_media_relation.sql 과 동등; media_items 는 본 레포 외 DDL)
-- ---------------------------------------------------------------------------
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
