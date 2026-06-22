-- =============================================================================
-- F-4.13 — schema_registry access_tier 컬럼 + general/medical 시드 (spec 040 Wave 1).
-- general 7키 키별 tier UPDATE · medical 7키 INSERT(general json_schema 복사, regulated).
-- 적용 순서: v280 이후.
-- =============================================================================

ALTER TABLE schema_registry
    ADD COLUMN IF NOT EXISTS access_tier VARCHAR(20) NOT NULL DEFAULT 'authenticated'
    CHECK (access_tier IN ('public', 'authenticated', 'authorized', 'regulated'));

UPDATE schema_registry SET access_tier = 'authenticated'
WHERE domain = 'general' AND meta_key IN ('summary', 'keywords', 'labels', 'objects', 'keyframes', 'caption');

UPDATE schema_registry SET access_tier = 'authorized'
WHERE domain = 'general' AND meta_key = 'stt';

INSERT INTO schema_registry (schema_id, domain, meta_key, json_schema, description, status, access_tier)
SELECT gen_random_uuid(), 'medical', meta_key, json_schema, description, status, 'regulated'
FROM schema_registry
WHERE domain = 'general'
ON CONFLICT (domain, meta_key) DO UPDATE SET
    json_schema = EXCLUDED.json_schema,
    description = EXCLUDED.description,
    status = EXCLUDED.status,
    access_tier = EXCLUDED.access_tier;
