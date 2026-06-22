-- =============================================================================
-- F-4.13 — ext_meta_field_registry 신설 + schema_registry backfill (spec 041).
-- schema_registry DDL 무터치(DROP/ALTER/RENAME 없음). OM 런타임 정본은 신규 테이블.
-- 이후 ext_meta 필드 시드는 ext_meta_field_registry만 대상(레거시 schema_registry 동기화 없음).
-- 선행: v290(access_tier on schema_registry). 적용 순서: v290 이후.
-- =============================================================================

CREATE TABLE IF NOT EXISTS ext_meta_field_registry (
    field_id     UUID PRIMARY KEY,
    domain       VARCHAR(50) NOT NULL,
    meta_key     VARCHAR(100) NOT NULL,
    json_schema  JSONB NOT NULL DEFAULT '{}'::jsonb,
    description  TEXT,
    status       VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'inactive')),
    access_tier  VARCHAR(20) NOT NULL DEFAULT 'authenticated'
        CHECK (access_tier IN ('public', 'authenticated', 'authorized', 'regulated')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_ext_meta_field_registry_domain_key UNIQUE (domain, meta_key)
);

COMMENT ON TABLE ext_meta_field_registry IS '도메인별 ext_meta 필드 정의 카탈로그(OM 정본).';
COMMENT ON COLUMN ext_meta_field_registry.field_id IS 'PK (UUIDv7).';

INSERT INTO ext_meta_field_registry (
    field_id, domain, meta_key, json_schema, description, status, access_tier, created_at
)
SELECT
    schema_id, domain, meta_key, json_schema, description, status, access_tier, created_at
FROM schema_registry
ON CONFLICT (domain, meta_key) DO NOTHING;
