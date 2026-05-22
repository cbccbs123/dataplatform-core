-- =============================================================================
-- F-4.1 통합 스키마 — Medical ER 6 테이블 (DDL 전용)
-- =============================================================================
-- 의료 엔티티 해소(ER) 7-Stage S-4~S-7 산출물의 저장소. 6월에는 DDL 만 생성하고
-- 적재 로직은 7~8월(F-5.5/5.6/5.8/5.9/5.10)에 구현한다.
-- entity_type 에는 옵션 B-7 의 'case' 를 미리 포함해 향후 ALTER 를 피한다.
-- 적용 순서: 110 은 100(core) 이후.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- entity: 9 의료 엔티티 + (옵션 B) case. promoted_to/promoted_at 는 B-7 Case→Patient 승격용.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity (
    entity_id   BIGSERIAL PRIMARY KEY,
    entity_type VARCHAR(20) NOT NULL
        CHECK (entity_type IN ('patient', 'visit', 'order', 'study', 'series',
                              'image', 'report', 'lab', 'diagnosis', 'case')),
    entity_uid  VARCHAR(255) NOT NULL,
    attributes  JSONB NOT NULL DEFAULT '{}'::jsonb,
    promoted_to BIGINT REFERENCES entity (entity_id) ON DELETE SET NULL,
    promoted_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_entity_type_uid UNIQUE (entity_type, entity_uid)
);

-- ---------------------------------------------------------------------------
-- asset_entity_link: asset ↔ entity N:N. decision='match' 시 적재.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS asset_entity_link (
    link_id    BIGSERIAL PRIMARY KEY,
    asset_id   BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    entity_id  BIGINT NOT NULL REFERENCES entity (entity_id) ON DELETE CASCADE,
    role       VARCHAR(50) NOT NULL DEFAULT '',
    confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_asset_entity_link UNIQUE (asset_id, entity_id, role)
);

-- ---------------------------------------------------------------------------
-- entity_edge: 9 엣지 타입(HAS_VISIT … HAS_STUDY). evidence_id 는 match_evidence 를 느슨 참조(FK 미설정).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity_edge (
    edge_id          BIGSERIAL PRIMARY KEY,
    source_entity_id BIGINT NOT NULL REFERENCES entity (entity_id) ON DELETE CASCADE,
    target_entity_id BIGINT NOT NULL REFERENCES entity (entity_id) ON DELETE CASCADE,
    edge_type        VARCHAR(30) NOT NULL
        CHECK (edge_type IN ('has_visit', 'has_order', 'orders_study', 'has_series',
                            'has_image', 'has_report', 'has_lab', 'has_diagnosis', 'has_study')),
    confidence       DOUBLE PRECISION,
    evidence_id      BIGINT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_entity_edge_no_self CHECK (source_entity_id <> target_entity_id),
    CONSTRAINT uq_entity_edge UNIQUE (source_entity_id, target_entity_id, edge_type)
);

-- ---------------------------------------------------------------------------
-- match_candidate: S-4 블로킹 결과(FS 스코어링 전 후보쌍). blocking_key 는 5종 키 중 하나.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_candidate (
    candidate_id   BIGSERIAL PRIMARY KEY,
    left_asset_id  BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    right_asset_id BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    blocking_key   VARCHAR(100) NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_match_candidate_no_self CHECK (left_asset_id <> right_asset_id),
    CONSTRAINT uq_match_candidate UNIQUE (left_asset_id, right_asset_id, blocking_key)
);

-- ---------------------------------------------------------------------------
-- match_decision: S-6 결정(score + match/review/non_match) + 정책 버전 스냅샷 + Negative Override.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_decision (
    decision_id       BIGSERIAL PRIMARY KEY,
    left_asset_id     BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    right_asset_id    BIGINT NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    score             DOUBLE PRECISION NOT NULL,
    decision          VARCHAR(20) NOT NULL
        CHECK (decision IN ('match', 'review', 'non_match')),
    negative_override BOOLEAN NOT NULL DEFAULT false,
    policy_version    VARCHAR(50) NOT NULL,
    decided_by        VARCHAR(20) NOT NULL DEFAULT 'auto'
        CHECK (decided_by IN ('auto', 'hitl')),
    reviewer_id       VARCHAR(100),
    decided_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_match_decision_no_self CHECK (left_asset_id <> right_asset_id)
);

-- ---------------------------------------------------------------------------
-- match_evidence: S-5 필드별 비교 증거(comparator·similarity·m·u·weight). 재현성 보장 핵심.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_evidence (
    evidence_id BIGSERIAL PRIMARY KEY,
    decision_id BIGINT NOT NULL REFERENCES match_decision (decision_id) ON DELETE CASCADE,
    field_name  VARCHAR(100) NOT NULL,
    comparator  VARCHAR(50) NOT NULL,
    similarity  DOUBLE PRECISION,
    m_prob      DOUBLE PRECISION,
    u_prob      DOUBLE PRECISION,
    weight      DOUBLE PRECISION,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
