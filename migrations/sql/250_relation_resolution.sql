-- =============================================================================
-- F-1.4 신뢰성 — 도메인-불가지 관계 재시도/미해소 큐 relation_resolution (spec 009 US3)
-- =============================================================================
-- cross-asset 관계 생성(run_relations)이 자산별로 관계를 못 만든 경우를 추적하는 경량 큐.
-- (a)일시적 실패(LLM/DB 예외)와 (b)고립(후보 없음·엣지 0)을 모두 'pending' 으로 흡수하고,
-- run_relations --retry 가 미해소(pending) + 미시도 자산만 골라 다시 시도한다. 상한(기본
-- relation_retry_max_attempts=3) 도달 시 'failed'(DLQ, 영구 격리)로 승격한다.
--
-- 설계 불변식(헌법 6·7조):
--   * asset.status FSM·CHECK 제약·ALLOWED_TRANSITIONS 는 무손상 — 관계 생성은 registered(terminal)
--     이후 단계이므로 별도 큐로 분리한다. 이 마이그레이션은 asset 계층을 건드리지 않는다.
--   * 도메인-불가지: 일반/의료/금융 어느 도메인이든 동일 흐름으로 pending 흡수. 의료 unresolved_pool
--     (단계 D)은 이 큐의 pending 부분집합으로 자연 흡수된다(별도 의료 전용 큐 미생성).
--   * 큐 페이로드(last_reason)는 예외 타입·고립 표식 등 비식별 사유만 — PHI/풀경로 미포함(헌법 10조).
--
-- PK 설계: asset_id 자연키 PK(자산당 1행 1:1 큐). surrogate UUIDv7 은 행을 2배 식별자로 만들 뿐이라
--   ON CONFLICT (asset_id) upsert·조회를 단순화하는 자연키를 채택한다(헌법 6조 PK 정책의 의도된 예외 —
--   자산/그래프 코어가 아닌 1:1 큐). FK 는 asset(asset_id) 참조.
--
-- ON DELETE CASCADE 근거: 큐 행은 자산에 종속된 파생 메타다. 자산이 삭제되면 그 자산의 미해소 큐
--   엔트리도 함께 사라져야 정합(고아 큐 행 방지). 삭제된 pending 은 --retry 선택에서 자연히 빠진다.
-- =============================================================================

CREATE TABLE IF NOT EXISTS relation_resolution (
    asset_id    UUID PRIMARY KEY REFERENCES asset (asset_id) ON DELETE CASCADE,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'resolved', 'failed')),
    attempts    INT NOT NULL DEFAULT 0,
    last_reason TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 부분 인덱스: --retry 는 미해소(pending)만 조회하므로 status='pending' 행만 색인한다.
-- 큐가 커져도 미해소 집합은 작게 유지되어 O(N_pending) 으로 골라낼 수 있다(resolved/failed 는 색인 제외).
CREATE INDEX IF NOT EXISTS idx_relation_resolution_pending
    ON relation_resolution (asset_id) WHERE status = 'pending';

COMMENT ON TABLE relation_resolution IS
    '도메인-불가지 관계 재시도/미해소 큐(자산당 1행). status=pending(일시 실패·고립) → resolved(엣지≥1) | failed(상한 도달 DLQ).';
COMMENT ON COLUMN relation_resolution.asset_id IS '자산 자연키 PK. FK→asset, ON DELETE CASCADE(자산 삭제 시 큐 행 동반 삭제).';
COMMENT ON COLUMN relation_resolution.status IS '큐 상태: pending(미해소) | resolved(관계 생성 성공) | failed(재시도 상한 도달, DLQ).';
COMMENT ON COLUMN relation_resolution.attempts IS '관계 생성 시도 누적 횟수. relation_retry_max_attempts 도달 시 failed 로 승격.';
COMMENT ON COLUMN relation_resolution.last_reason IS '마지막 미해소 사유(비식별: 예외 타입·고립 표식만, PHI/풀경로 미포함 — 헌법 10조).';
