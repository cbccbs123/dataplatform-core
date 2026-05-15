-- Upgrade: relation_topic.status 제거, relation_type.status 기본값 inactive.
DROP INDEX IF EXISTS idx_relation_topic_status;

ALTER TABLE relation_topic
    DROP COLUMN IF EXISTS status;

ALTER TABLE relation_type
    ALTER COLUMN status SET DEFAULT 'inactive';
