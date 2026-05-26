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

-- =============================================================================
-- 테이블·컬럼 설명 (COMMENT) — 의료 ER (S-4~S-7, 적재 로직은 7~8월)
-- =============================================================================

COMMENT ON TABLE entity IS '의료 엔티티(9종 + 옵션 B case). ER 해소 결과의 정규 엔티티.';
COMMENT ON COLUMN entity.entity_id   IS 'PK.';
COMMENT ON COLUMN entity.entity_type IS '엔티티 종류: patient/visit/order/study/series/image/report/lab/diagnosis/case.';
COMMENT ON COLUMN entity.entity_uid  IS '정규화된 식별자 또는 hash(path+arrival)(case).';
COMMENT ON COLUMN entity.attributes  IS '엔티티 속성(jsonb).';
COMMENT ON COLUMN entity.promoted_to IS '옵션 B-7 승격 대상 entity_id(case→patient). 승격 시 채움.';
COMMENT ON COLUMN entity.promoted_at IS '승격 시각.';
COMMENT ON COLUMN entity.created_at  IS '생성 시각.';

COMMENT ON TABLE asset_entity_link IS '자산↔엔티티 N:N 연결. decision=match 시 적재.';
COMMENT ON COLUMN asset_entity_link.link_id    IS 'PK.';
COMMENT ON COLUMN asset_entity_link.asset_id   IS 'FK→asset.';
COMMENT ON COLUMN asset_entity_link.entity_id  IS 'FK→entity.';
COMMENT ON COLUMN asset_entity_link.role       IS '연결 역할.';
COMMENT ON COLUMN asset_entity_link.confidence IS '연결 신뢰도.';
COMMENT ON COLUMN asset_entity_link.created_at IS '생성 시각.';

COMMENT ON TABLE entity_edge IS '엔티티 간 관계 엣지(9종: HAS_VISIT … HAS_STUDY).';
COMMENT ON COLUMN entity_edge.edge_id          IS 'PK.';
COMMENT ON COLUMN entity_edge.source_entity_id IS 'FK→entity(출발).';
COMMENT ON COLUMN entity_edge.target_entity_id IS 'FK→entity(도착).';
COMMENT ON COLUMN entity_edge.edge_type        IS '엣지 종류: has_visit/has_order/orders_study/has_series/has_image/has_report/has_lab/has_diagnosis/has_study.';
COMMENT ON COLUMN entity_edge.confidence       IS '엣지 신뢰도.';
COMMENT ON COLUMN entity_edge.evidence_id      IS 'match_evidence 참조(느슨한 연결, FK 미설정).';
COMMENT ON COLUMN entity_edge.created_at       IS '생성 시각.';

COMMENT ON TABLE match_candidate IS 'S-4 블로킹 결과 후보쌍(FS 스코어링 전).';
COMMENT ON COLUMN match_candidate.candidate_id   IS 'PK.';
COMMENT ON COLUMN match_candidate.left_asset_id  IS 'FK→asset(좌).';
COMMENT ON COLUMN match_candidate.right_asset_id IS 'FK→asset(우).';
COMMENT ON COLUMN match_candidate.blocking_key   IS '묶인 블로킹 키(5종 중).';
COMMENT ON COLUMN match_candidate.created_at     IS '생성 시각.';

COMMENT ON TABLE match_decision IS 'S-6 매칭 결정(score + match/review/non_match) + 정책 스냅샷.';
COMMENT ON COLUMN match_decision.decision_id       IS 'PK.';
COMMENT ON COLUMN match_decision.left_asset_id     IS 'FK→asset(좌).';
COMMENT ON COLUMN match_decision.right_asset_id    IS 'FK→asset(우).';
COMMENT ON COLUMN match_decision.score             IS 'Fellegi-Sunter 가중합 점수.';
COMMENT ON COLUMN match_decision.decision          IS '결정: match|review|non_match.';
COMMENT ON COLUMN match_decision.negative_override IS 'Negative Override 발동 여부(강한 불일치 거부).';
COMMENT ON COLUMN match_decision.policy_version    IS 'er_policy.threshold_v 스냅샷.';
COMMENT ON COLUMN match_decision.decided_by        IS '결정 주체: auto|hitl.';
COMMENT ON COLUMN match_decision.reviewer_id       IS 'HITL 검토자 식별.';
COMMENT ON COLUMN match_decision.decided_at        IS '결정 시각.';

COMMENT ON TABLE match_evidence IS 'S-5 필드별 비교 증거(재현성 핵심).';
COMMENT ON COLUMN match_evidence.evidence_id IS 'PK.';
COMMENT ON COLUMN match_evidence.decision_id IS 'FK→match_decision.';
COMMENT ON COLUMN match_evidence.field_name  IS '비교 필드명.';
COMMENT ON COLUMN match_evidence.comparator  IS '비교기: exact/jaro_winkler/embedding_cosine 등.';
COMMENT ON COLUMN match_evidence.similarity  IS '유사도 값.';
COMMENT ON COLUMN match_evidence.m_prob      IS 'Fellegi-Sunter m 확률(일치 시).';
COMMENT ON COLUMN match_evidence.u_prob      IS 'Fellegi-Sunter u 확률(우연 일치).';
COMMENT ON COLUMN match_evidence.weight      IS '필드 가중치 log(m/u).';
COMMENT ON COLUMN match_evidence.created_at  IS '생성 시각.';
