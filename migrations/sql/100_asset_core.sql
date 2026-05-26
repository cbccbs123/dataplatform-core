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

-- =============================================================================
-- 테이블·컬럼 설명 (COMMENT)
-- =============================================================================

COMMENT ON TABLE asset_group IS '자산 묶음. study_uid/MRN 등 그룹 키로 여러 asset 을 하나로 묶는다(의료 그룹화 등).';
COMMENT ON COLUMN asset_group.group_id   IS 'PK. 그룹 식별자.';
COMMENT ON COLUMN asset_group.group_key  IS '그룹 키 값(예: study_uid 문자열, MRN).';
COMMENT ON COLUMN asset_group.group_kind IS '그룹 키 종류: general|study_uid|mrn|manual.';
COMMENT ON COLUMN asset_group.created_at IS '생성 시각.';

COMMENT ON TABLE asset IS '원본 파일 단위 자산. 수집된 파일 1건당 1행. media_items 를 대체한다.';
COMMENT ON COLUMN asset.asset_id          IS 'PK. 자산 식별자.';
COMMENT ON COLUMN asset.group_id          IS 'FK→asset_group. 소속 그룹(없으면 NULL).';
COMMENT ON COLUMN asset.modality          IS '모달리티(MediaKind/OfficeKind 값): txt/pdf/json/word/excel/powerpoint/image/video/audio/unknown.';
COMMENT ON COLUMN asset.fs_path           IS '파일 시스템 경로.';
COMMENT ON COLUMN asset.fs_uri            IS '원본 URI(선택).';
COMMENT ON COLUMN asset.file_hash         IS '파일 해시(예: SHA-256). 중복 판별용(선택).';
COMMENT ON COLUMN asset.file_size         IS '파일 크기(byte).';
COMMENT ON COLUMN asset.domain_label      IS '도메인 분류 결과: medical|general|review.';
COMMENT ON COLUMN asset.domain_confidence IS '도메인 분류 신뢰도(0~1).';
COMMENT ON COLUMN asset.status            IS '처리 상태머신: received→routing→classifying→extracting→registered, 실패 시 failed.';
COMMENT ON COLUMN asset.status_reason     IS '실패/상태 부가 사유.';
COMMENT ON COLUMN asset.created_at        IS '생성(수신) 시각.';
COMMENT ON COLUMN asset.updated_at        IS '마지막 갱신 시각.';

COMMENT ON TABLE asset_metadata IS '자산 메타데이터(2-layer). 자산당 1행.';
COMMENT ON COLUMN asset_metadata.asset_id      IS 'PK·FK→asset. 1:1.';
COMMENT ON COLUMN asset_metadata.core_meta     IS '파일/시스템 메타(jsonb): 크기·인코딩·해상도·언어·페이지수 등.';
COMMENT ON COLUMN asset_metadata.ext_meta      IS '도메인 신호(jsonb): summary·keywords·labels·objects·keyframes·stt·caption 등.';
COMMENT ON COLUMN asset_metadata.tags          IS '태그 배열.';
COMMENT ON COLUMN asset_metadata.search_vector IS 'FTS tsvector(메타 평문 기반).';
COMMENT ON COLUMN asset_metadata.created_at    IS '생성 시각.';
COMMENT ON COLUMN asset_metadata.updated_at    IS '마지막 갱신 시각.';

COMMENT ON TABLE asset_embedding IS '자산 임베딩 벡터. 채널·청크별 1536D. PK=(asset_id, channel, chunk_index).';
COMMENT ON COLUMN asset_embedding.asset_id      IS 'FK→asset.';
COMMENT ON COLUMN asset_embedding.channel       IS '임베딩 채널: st(텍스트)·clip(시각)·(옵션 B) medclip 등.';
COMMENT ON COLUMN asset_embedding.chunk_index   IS '청크/키프레임 인덱스(다청크·다프레임 수용).';
COMMENT ON COLUMN asset_embedding.embedding     IS 'pgvector 1536D 벡터.';
COMMENT ON COLUMN asset_embedding.model_name    IS '임베딩 모델명.';
COMMENT ON COLUMN asset_embedding.model_version IS '임베딩 모델 버전(선택).';
COMMENT ON COLUMN asset_embedding.created_at    IS '생성 시각.';

COMMENT ON TABLE asset_lineage IS 'PROV-DM(W3C 2013) 활동 로그. 결정 재현성·데이터 계보 추적.';
COMMENT ON COLUMN asset_lineage.lineage_id  IS 'PK.';
COMMENT ON COLUMN asset_lineage.asset_id    IS 'FK→asset.';
COMMENT ON COLUMN asset_lineage.activity    IS '활동 식별(예: extract.text.v1, classify.cascade.v1, case_promotion.v1).';
COMMENT ON COLUMN asset_lineage.agent       IS '수행 주체(모듈/모델/사용자).';
COMMENT ON COLUMN asset_lineage.used        IS 'prov:used 입력 엔티티(jsonb).';
COMMENT ON COLUMN asset_lineage.generated   IS 'prov:wasGeneratedBy 산출물(jsonb).';
COMMENT ON COLUMN asset_lineage.payload     IS '부가 정보(jsonb).';
COMMENT ON COLUMN asset_lineage.occurred_at IS '활동 시각.';

COMMENT ON TABLE access_log IS 'API 호출/접근 이력(F-4.6).';
COMMENT ON COLUMN access_log.access_id   IS 'PK.';
COMMENT ON COLUMN access_log.asset_id    IS 'FK→asset(없을 수 있음).';
COMMENT ON COLUMN access_log.user_id     IS '호출 사용자 식별.';
COMMENT ON COLUMN access_log.action      IS '동작: search/view/download/api 등.';
COMMENT ON COLUMN access_log.detail      IS '부가 정보(jsonb).';
COMMENT ON COLUMN access_log.occurred_at IS '발생 시각.';

COMMENT ON TABLE schema_registry IS '도메인별 ext_meta 키 정의 + JSON Schema 검증 규칙.';
COMMENT ON COLUMN schema_registry.schema_id   IS 'PK.';
COMMENT ON COLUMN schema_registry.domain      IS '도메인: medical/general 등.';
COMMENT ON COLUMN schema_registry.meta_key    IS 'ext_meta 키 이름.';
COMMENT ON COLUMN schema_registry.json_schema IS '해당 키의 JSON Schema(jsonb).';
COMMENT ON COLUMN schema_registry.description IS '설명.';
COMMENT ON COLUMN schema_registry.status      IS '활성 상태: active|inactive.';
COMMENT ON COLUMN schema_registry.created_at  IS '생성 시각.';

COMMENT ON TABLE asset_classification IS 'F-5.1 3-stage cascade 도메인 분류 결과. 자산당 1행.';
COMMENT ON COLUMN asset_classification.classification_id IS 'PK.';
COMMENT ON COLUMN asset_classification.asset_id          IS 'FK→asset(유니크).';
COMMENT ON COLUMN asset_classification.stage1_scores     IS 'Stage1(magic-byte/경로) 점수(jsonb).';
COMMENT ON COLUMN asset_classification.stage2_scores     IS 'Stage2(CLIP/ST zero-shot) 점수(jsonb).';
COMMENT ON COLUMN asset_classification.stage3_scores     IS 'Stage3(온프레미스 LLM zero-shot) 점수(jsonb).';
COMMENT ON COLUMN asset_classification.final_label       IS '최종 라벨: medical|general|review.';
COMMENT ON COLUMN asset_classification.confidence        IS '최종 신뢰도(0~1).';
COMMENT ON COLUMN asset_classification.decided_stage     IS '확정 단계: 1|2|3.';
COMMENT ON COLUMN asset_classification.policy_version    IS '분류 정책 버전 스냅샷.';
COMMENT ON COLUMN asset_classification.created_at        IS '생성 시각.';
