-- =============================================================================
-- F-4.1 통합 스키마 — Core 8 테이블
-- =============================================================================
-- 신규 asset_* 통합 스키마(2026 2차년도). 기존 media_items/media_chunks 를 대체한다.
-- enum 류는 기존 마이그레이션 관례대로 네이티브 ENUM 대신 VARCHAR + CHECK 로 표현한다.
-- 적용 순서: 100(core) → 110(medical_er) → 120(operational) → 130(indexes).
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- asset_group: study_uid / MRN 등 그룹 키로 묶는 자산 묶음 (asset 보다 먼저 생성)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asset_group (
    group_id   BIGSERIAL PRIMARY KEY,
    group_key  VARCHAR(255) NOT NULL,
    group_kind VARCHAR(50) NOT NULL DEFAULT 'general'
        CHECK (group_kind IN ('general', 'study_uid', 'mrn', 'manual')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_asset_group_kind_key UNIQUE (group_kind, group_key)
);

-- ---------------------------------------------------------------------------
-- asset: 원본 파일 단위. modality 값은 MediaKind/OfficeKind 값과 일치한다.
-- status 는 F-1.4 상태 머신: received → routing → classifying → extracting → registered (실패 시 failed)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asset (
    asset_id          BIGSERIAL PRIMARY KEY,
    group_id          BIGINT REFERENCES asset_group (group_id) ON DELETE SET NULL,
    modality          VARCHAR(20) NOT NULL
        CHECK (modality IN ('txt', 'pdf', 'json', 'word', 'excel', 'powerpoint',
                            'image', 'video', 'audio', 'unknown')),
    fs_path           TEXT NOT NULL,
    fs_uri            TEXT,
    file_hash         VARCHAR(64),
    file_size         BIGINT,
    domain_label      VARCHAR(20) NOT NULL DEFAULT 'general'
        CHECK (domain_label IN ('medical', 'general', 'review')),
    domain_confidence DOUBLE PRECISION,
    status            VARCHAR(20) NOT NULL DEFAULT 'received'
        CHECK (status IN ('received', 'routing', 'classifying', 'extracting', 'registered', 'failed')),
    status_reason     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- asset_metadata: 2-layer 메타. core_meta = 파일/시스템 메타, ext_meta = 도메인 신호(요약·키워드·labels 등).
-- search_vector 는 FTS 평문(media_item_search_text.build_media_item_fts_plain 결과) 기반.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asset_metadata (
    asset_id      BIGINT PRIMARY KEY REFERENCES asset (asset_id) ON DELETE CASCADE,
    core_meta     JSONB NOT NULL DEFAULT '{}'::jsonb,
    ext_meta      JSONB NOT NULL DEFAULT '{}'::jsonb,
    tags          TEXT[] NOT NULL DEFAULT '{}',
    search_vector TSVECTOR,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- asset_embedding: 채널(st/clip/…)·청크별 1536D 벡터. 텍스트 다청크·영상 키프레임 다건을 수용하려 chunk_index 포함.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asset_embedding (
    asset_id      BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    channel       VARCHAR(20) NOT NULL,
    chunk_index   INTEGER NOT NULL DEFAULT 0,
    embedding     VECTOR(1536) NOT NULL,
    model_name    VARCHAR(200) NOT NULL,
    model_version VARCHAR(100),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT pk_asset_embedding PRIMARY KEY (asset_id, channel, chunk_index)
);

-- ---------------------------------------------------------------------------
-- asset_lineage: PROV-DM(W3C 2013) Entity-Activity-Agent 활동 로그. 결정 재현성·계보 추적.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asset_lineage (
    lineage_id  BIGSERIAL PRIMARY KEY,
    asset_id    BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    activity    VARCHAR(100) NOT NULL,
    agent       VARCHAR(100) NOT NULL,
    used        JSONB NOT NULL DEFAULT '{}'::jsonb,
    generated   JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- access_log: 모든 API 호출 이력(F-4.6).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS access_log (
    access_id   BIGSERIAL PRIMARY KEY,
    asset_id    BIGINT REFERENCES asset (asset_id) ON DELETE SET NULL,
    user_id     VARCHAR(100),
    action      VARCHAR(50) NOT NULL,
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- schema_registry: 도메인별 ext_meta 키 정의 + JSON Schema 검증 규칙.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_registry (
    schema_id   BIGSERIAL PRIMARY KEY,
    domain      VARCHAR(50) NOT NULL,
    meta_key    VARCHAR(100) NOT NULL,
    json_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    description TEXT,
    status      VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'inactive')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_schema_registry_domain_key UNIQUE (domain, meta_key)
);

-- ---------------------------------------------------------------------------
-- asset_classification: F-5.1 3-stage cascade 분류 결과(단계별 점수 + 최종 라벨).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asset_classification (
    classification_id BIGSERIAL PRIMARY KEY,
    asset_id          BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    stage1_scores     JSONB NOT NULL DEFAULT '{}'::jsonb,
    stage2_scores     JSONB NOT NULL DEFAULT '{}'::jsonb,
    stage3_scores     JSONB NOT NULL DEFAULT '{}'::jsonb,
    final_label       VARCHAR(20) NOT NULL
        CHECK (final_label IN ('medical', 'general', 'review')),
    confidence        DOUBLE PRECISION,
    decided_stage     SMALLINT
        CHECK (decided_stage IN (1, 2, 3)),
    policy_version    VARCHAR(50),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_asset_classification_asset UNIQUE (asset_id)
);
