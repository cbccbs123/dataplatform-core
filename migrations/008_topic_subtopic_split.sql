-- Split flat relation_topic into relation_topic_parent (topic_ko, topic_en) +
-- relation_subtopic (topic_id, subtopic_ko, subtopic_en). relation_type references relation_subtopic_id.
-- Requires: 004 (relation_topic, relation_type). Run once per environment; re-run skips destructive steps if already applied.

-- ---------------------------------------------------------------------------
-- 1. New tables
-- ---------------------------------------------------------------------------
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

-- ---------------------------------------------------------------------------
-- 2. Backfill from legacy relation_topic (only while legacy table exists)
-- ---------------------------------------------------------------------------
DO $body$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'relation_topic'
    ) THEN
        INSERT INTO relation_topic_parent (topic_ko, topic_en)
        SELECT DISTINCT topic_ko, topic_en FROM relation_topic
        ON CONFLICT (topic_ko, topic_en) DO NOTHING;

        INSERT INTO relation_subtopic (topic_id, subtopic_ko, subtopic_en)
        SELECT DISTINCT p.topic_id, r.subtopic_ko, r.subtopic_en
        FROM relation_topic r
        JOIN relation_topic_parent p
          ON p.topic_ko = r.topic_ko AND p.topic_en = r.topic_en
        ON CONFLICT (topic_id, subtopic_ko, subtopic_en) DO NOTHING;
    END IF;
END
$body$;

-- ---------------------------------------------------------------------------
-- 3. relation_type: add relation_subtopic_id, backfill, swap FK (only if legacy column remains)
-- ---------------------------------------------------------------------------
ALTER TABLE relation_type ADD COLUMN IF NOT EXISTS relation_subtopic_id BIGINT;

DO $swap$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'relation_type'
          AND column_name = 'relation_topic_id'
    ) THEN
        UPDATE relation_type rt
        SET relation_subtopic_id = s.subtopic_id
        FROM relation_topic r
        JOIN relation_topic_parent p ON p.topic_ko = r.topic_ko AND p.topic_en = r.topic_en
        JOIN relation_subtopic s
          ON s.topic_id = p.topic_id
         AND s.subtopic_ko = r.subtopic_ko
         AND s.subtopic_en = r.subtopic_en
        WHERE r.relation_topic_id = rt.relation_topic_id
          AND rt.relation_subtopic_id IS NULL;

        ALTER TABLE relation_type ALTER COLUMN relation_subtopic_id SET NOT NULL;

        ALTER TABLE relation_type DROP CONSTRAINT IF EXISTS relation_type_relation_topic_id_fkey;
        ALTER TABLE relation_type DROP CONSTRAINT IF EXISTS uq_relation_type_kind_topic;
        ALTER TABLE relation_type DROP COLUMN relation_topic_id;

        ALTER TABLE relation_type
            ADD CONSTRAINT relation_type_relation_subtopic_id_fkey
            FOREIGN KEY (relation_subtopic_id) REFERENCES relation_subtopic (subtopic_id) ON DELETE RESTRICT;

        ALTER TABLE relation_type
            ADD CONSTRAINT uq_relation_type_kind_subtopic UNIQUE (relation_kind_id, relation_subtopic_id);

        DROP INDEX IF EXISTS idx_relation_type_topic;
        CREATE INDEX IF NOT EXISTS idx_relation_type_subtopic ON relation_type (relation_subtopic_id);

        DROP TABLE IF EXISTS relation_topic;
    END IF;
END
$swap$;

-- If swap block was skipped (already migrated) but indexes/constraints missing, repair (best-effort).
DO $repair$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'relation_type'
          AND column_name = 'relation_subtopic_id'
    )
       AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'relation_type'
          AND column_name = 'relation_topic_id'
    ) THEN
        BEGIN
            ALTER TABLE relation_type
                ADD CONSTRAINT relation_type_relation_subtopic_id_fkey
                FOREIGN KEY (relation_subtopic_id) REFERENCES relation_subtopic (subtopic_id) ON DELETE RESTRICT;
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END;
        BEGIN
            ALTER TABLE relation_type
                ADD CONSTRAINT uq_relation_type_kind_subtopic UNIQUE (relation_kind_id, relation_subtopic_id);
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END;
        CREATE INDEX IF NOT EXISTS idx_relation_type_subtopic ON relation_type (relation_subtopic_id);
    END IF;
END
$repair$;
