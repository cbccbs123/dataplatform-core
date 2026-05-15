-- Relations schema (single migration): relation_kind, relation_topic, relation_type (FK bundle),
-- media_relation, relation_proposal. Requires: media_items (existing pipeline).
-- relation_type: no type_name/description; status defaults to inactive (seeded catalog rows explicit active).
-- relation_topic: no status column.

-- ---------------------------------------------------------------------------
-- relation_kind: coarse relation class (same_domain, duplicate_near, …)
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

CREATE INDEX IF NOT EXISTS idx_relation_kind_status ON relation_kind (status);

-- ---------------------------------------------------------------------------
-- relation_topic: normalized topic tuple (KO/EN + subtopics)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS relation_topic (
    relation_topic_id BIGSERIAL PRIMARY KEY,
    topic_ko          VARCHAR(200) NOT NULL DEFAULT '',
    subtopic_ko       VARCHAR(200) NOT NULL DEFAULT '',
    topic_en          VARCHAR(200) NOT NULL DEFAULT '',
    subtopic_en       VARCHAR(200) NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_relation_topic_koen UNIQUE (topic_ko, subtopic_ko, topic_en, subtopic_en)
);

CREATE INDEX IF NOT EXISTS idx_relation_topic_topic_ko ON relation_topic (topic_ko);
CREATE INDEX IF NOT EXISTS idx_relation_topic_topic_en ON relation_topic (topic_en);

-- ---------------------------------------------------------------------------
-- relation_type: one row per (kind, topic) bundle; edges reference this only
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS relation_type (
    relation_type_id   BIGSERIAL PRIMARY KEY,
    relation_kind_id  BIGINT NOT NULL REFERENCES relation_kind (relation_kind_id) ON DELETE RESTRICT,
    relation_topic_id  BIGINT NOT NULL REFERENCES relation_topic (relation_topic_id) ON DELETE RESTRICT,
    status              VARCHAR(20) NOT NULL DEFAULT 'inactive'
        CHECK (status IN ('active', 'inactive')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_relation_type_kind_topic UNIQUE (relation_kind_id, relation_topic_id)
);

CREATE INDEX IF NOT EXISTS idx_relation_type_status ON relation_type (status);
CREATE INDEX IF NOT EXISTS idx_relation_type_kind ON relation_type (relation_kind_id);
CREATE INDEX IF NOT EXISTS idx_relation_type_topic ON relation_type (relation_topic_id);

-- ---------------------------------------------------------------------------
-- media_relation
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS media_relation (
    relation_id           BIGSERIAL PRIMARY KEY,
    source_media_item_id BIGINT NOT NULL REFERENCES media_items (id) ON DELETE CASCADE,
    target_media_item_id  BIGINT REFERENCES media_items (id) ON DELETE SET NULL,
    target_ref            TEXT NOT NULL,
    relation_type_id      BIGINT NOT NULL REFERENCES relation_type (relation_type_id) ON DELETE RESTRICT,
    confidence            NUMERIC(5, 4),
    basis                 TEXT,
    evidence_json         JSONB NOT NULL DEFAULT '{}'::jsonb,
    generation_method     VARCHAR(20) NOT NULL DEFAULT 'llm'
        CHECK (generation_method IN ('rule', 'llm', 'hybrid')),
    status                VARCHAR(20) NOT NULL DEFAULT 'confirmed'
        CHECK (status IN ('confirmed', 'proposed')),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_media_relation_edge UNIQUE (source_media_item_id, target_ref, relation_type_id)
);

CREATE INDEX IF NOT EXISTS idx_media_relation_source ON media_relation (source_media_item_id);
CREATE INDEX IF NOT EXISTS idx_media_relation_target ON media_relation (target_media_item_id);
CREATE INDEX IF NOT EXISTS idx_media_relation_type ON media_relation (relation_type_id);

-- ---------------------------------------------------------------------------
-- relation_proposal
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS relation_proposal (
    candidate_id           BIGSERIAL PRIMARY KEY,
    source_media_item_id   BIGINT NOT NULL REFERENCES media_items (id) ON DELETE CASCADE,
    target_media_item_id   BIGINT REFERENCES media_items (id) ON DELETE SET NULL,
    target_ref             TEXT,
    relation_id            BIGINT REFERENCES media_relation (relation_id) ON DELETE SET NULL,
    relation_type_id       BIGINT REFERENCES relation_type (relation_type_id) ON DELETE SET NULL,
    suggested_type_code    VARCHAR(100),
    confidence             NUMERIC(5, 4),
    reason                 TEXT,
    validation_failure     VARCHAR(64) NOT NULL,
    llm_output             JSONB,
    status                 VARCHAR(32) NOT NULL DEFAULT 'pending_review',
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_relation_proposal_source ON relation_proposal (source_media_item_id);
CREATE INDEX IF NOT EXISTS idx_relation_proposal_status ON relation_proposal (status);
CREATE INDEX IF NOT EXISTS idx_relation_proposal_relation_id ON relation_proposal (relation_id);

-- ---------------------------------------------------------------------------
-- Seeds: kinds + default topic + relation_type rows (kind × default topic, active for LLM catalog)
-- ---------------------------------------------------------------------------
INSERT INTO relation_kind (kind_code, kind_name_ko, description, is_symmetric, status)
VALUES
    ('same_domain', '동일 도메인', '주제·분야가 같은 연결', true, 'active'),
    ('same_series', '동일 시리즈', '같은 시리즈·연작·라인업 연결', true, 'active'),
    ('duplicate_near', '유사 근접', '내용/주제/장면의 근접 유사', true, 'active'),
    ('references', '참조', '명시적 인용·링크·제목 참조', false, 'active'),
    ('derived_from', '파생', '한 콘텐츠가 다른 콘텐츠에서 파생', false, 'active')
ON CONFLICT (kind_code) DO NOTHING;

INSERT INTO relation_topic (topic_ko, subtopic_ko, topic_en, subtopic_en)
VALUES ('일반', '', 'general', '')
ON CONFLICT (topic_ko, subtopic_ko, topic_en, subtopic_en) DO NOTHING;

INSERT INTO relation_type (relation_kind_id, relation_topic_id, status)
SELECT k.relation_kind_id,
       t.relation_topic_id,
       'active'
FROM relation_kind k
CROSS JOIN relation_topic t
WHERE t.topic_ko = '일반'
  AND t.subtopic_ko = ''
  AND t.topic_en = 'general'
  AND t.subtopic_en = ''
  AND k.kind_code IN (
      'same_domain',
      'same_series',
      'duplicate_near',
      'references',
      'derived_from'
  )
  AND k.status = 'active'
ON CONFLICT (relation_kind_id, relation_topic_id) DO NOTHING;
