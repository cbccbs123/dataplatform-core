-- =============================================================================
-- F-1.3 보강 — 자산 내용 기반 중복 방지(부분 유니크 인덱스)
-- =============================================================================
-- registered 상태 자산은 동일 file_hash 를 중복 가질 수 없다(내용 dedup).
-- received/failed 등 비종료/실패 상태나 file_hash NULL 은 제약 대상 외.
-- 앱(run_ingest)이 적재 전 사전 조회로 skip 하며, 본 인덱스는 경쟁 상황 안전망.
-- =============================================================================

CREATE UNIQUE INDEX IF NOT EXISTS uq_asset_file_hash_registered
    ON asset (file_hash)
    WHERE status = 'registered' AND file_hash IS NOT NULL;

COMMENT ON INDEX uq_asset_file_hash_registered IS
    'registered 자산은 동일 file_hash 중복 불가(내용 기반 dedup 안전망).';
