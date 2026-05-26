-- =============================================================================
-- F-4.1 통합 스키마 — 운영 3 테이블 + 옵션 B content_cluster 1 테이블
-- =============================================================================
-- review_queue(HITL 대기), unresolved_pool(미해소 풀), er_policy(정책 버전).
-- content_cluster 는 옵션 B-4(HDBSCAN 콘텐츠 클러스터) 저장소.
-- 적용 순서: 120 은 110(medical_er) 이후(review_queue 가 match_decision 을 참조).
-- =============================================================================

-- ---------------------------------------------------------------------------
-- review_queue: HITL 검토 대기. 매치 결정쌍(decision_id) 또는 애매한 분류 자산(asset_id) 모두 수용.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_queue (
    queue_id       UUID PRIMARY KEY,
    decision_id    UUID REFERENCES match_decision (decision_id) ON DELETE CASCADE,
    asset_id       UUID REFERENCES asset (asset_id) ON DELETE CASCADE,
    priority_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    status         VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_review', 'resolved')),
    payload        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at    TIMESTAMPTZ
);

-- ---------------------------------------------------------------------------
-- unresolved_pool: non_match·미지원 모달리티·저신뢰 자산. 신규 자산/정책 갱신 시 재블로킹 대상.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS unresolved_pool (
    pool_id       UUID PRIMARY KEY,
    asset_id      UUID NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    reason        VARCHAR(100) NOT NULL,
    reblock_after TIMESTAMPTZ,
    payload       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_unresolved_pool_asset UNIQUE (asset_id)
);

-- ---------------------------------------------------------------------------
-- er_policy: ER 정책 버전 스냅샷(threshold_v). T_match=10.0, T_review=4.0 기본.
-- mu_config 는 비교기별 고정 m·u, extra 는 Negative Override·time_window 등.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS er_policy (
    policy_id   UUID PRIMARY KEY,
    threshold_v VARCHAR(50) NOT NULL UNIQUE,
    t_match     DOUBLE PRECISION NOT NULL DEFAULT 10.0,
    t_review    DOUBLE PRECISION NOT NULL DEFAULT 4.0,
    mu_config   JSONB NOT NULL DEFAULT '{}'::jsonb,
    extra       JSONB NOT NULL DEFAULT '{}'::jsonb,
    status      VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'inactive')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- content_cluster: 옵션 B-4 HDBSCAN 클러스터 id(채널별). cluster_id=-1 은 noise.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS content_cluster (
    cluster_row_id UUID PRIMARY KEY,
    asset_id       UUID NOT NULL REFERENCES asset (asset_id) ON DELETE CASCADE,
    channel        VARCHAR(20) NOT NULL
        CHECK (channel IN ('visual', 'text')),
    cluster_id     INTEGER NOT NULL,
    model_name     VARCHAR(200),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_content_cluster_asset_channel UNIQUE (asset_id, channel)
);

-- =============================================================================
-- 테이블·컬럼 설명 (COMMENT)
-- =============================================================================

COMMENT ON TABLE review_queue IS 'HITL 검토 대기열. 매칭 결정쌍 또는 애매한 자산.';
COMMENT ON COLUMN review_queue.queue_id       IS 'PK.';
COMMENT ON COLUMN review_queue.decision_id    IS 'FK→match_decision(매칭 검토 시).';
COMMENT ON COLUMN review_queue.asset_id        IS 'FK→asset(자산 단위 검토 시).';
COMMENT ON COLUMN review_queue.priority_score IS '우선순위 점수(높을수록 먼저).';
COMMENT ON COLUMN review_queue.status         IS '상태: pending|in_review|resolved.';
COMMENT ON COLUMN review_queue.payload        IS '검토 부가 정보(jsonb).';
COMMENT ON COLUMN review_queue.created_at     IS '생성 시각.';
COMMENT ON COLUMN review_queue.resolved_at    IS '해결 시각.';

COMMENT ON TABLE unresolved_pool IS '미해소 풀. non_match/미지원 모달리티/저신뢰 자산. 재블로킹 대상.';
COMMENT ON COLUMN unresolved_pool.pool_id       IS 'PK.';
COMMENT ON COLUMN unresolved_pool.asset_id      IS 'FK→asset(유니크).';
COMMENT ON COLUMN unresolved_pool.reason        IS '사유: non_match/unknown_modality/low_confidence 등.';
COMMENT ON COLUMN unresolved_pool.reblock_after IS '재블로킹 트리거 시점(선택).';
COMMENT ON COLUMN unresolved_pool.payload       IS '부가 정보(jsonb).';
COMMENT ON COLUMN unresolved_pool.created_at    IS '생성 시각.';

COMMENT ON TABLE er_policy IS 'ER 정책 버전 스냅샷. 결정 재현성 보장.';
COMMENT ON COLUMN er_policy.policy_id   IS 'PK.';
COMMENT ON COLUMN er_policy.threshold_v IS '정책 버전 식별자(유니크).';
COMMENT ON COLUMN er_policy.t_match     IS 'T_match 임계(기본 10.0).';
COMMENT ON COLUMN er_policy.t_review    IS 'T_review 임계(기본 4.0).';
COMMENT ON COLUMN er_policy.mu_config   IS '비교기별 고정 m·u(jsonb).';
COMMENT ON COLUMN er_policy.extra       IS 'Negative Override·time_window 등(jsonb).';
COMMENT ON COLUMN er_policy.status      IS '활성 상태: active|inactive.';
COMMENT ON COLUMN er_policy.created_at  IS '생성 시각.';

COMMENT ON TABLE content_cluster IS '옵션 B-4 HDBSCAN 콘텐츠 클러스터(채널별).';
COMMENT ON COLUMN content_cluster.cluster_row_id IS 'PK.';
COMMENT ON COLUMN content_cluster.asset_id       IS 'FK→asset.';
COMMENT ON COLUMN content_cluster.channel        IS '채널: visual|text.';
COMMENT ON COLUMN content_cluster.cluster_id     IS 'HDBSCAN 클러스터 id(-1=noise).';
COMMENT ON COLUMN content_cluster.model_name     IS '클러스터링 사용 모델명.';
COMMENT ON COLUMN content_cluster.created_at     IS '생성 시각.';
