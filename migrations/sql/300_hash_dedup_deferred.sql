-- =============================================================================
-- 069 US-B FR-B11(P2-8) — 해시 dedup 유니크 인덱스를 deferred 까지 확장
-- =============================================================================
-- 문제: 기존 uq_asset_file_hash_registered(150)는 status='registered' 만 커버 —
--   앱 dedup 사전조회(run_ingest)는 registered+deferred 를 보는데(009 확장), DB 안전망은
--   registered 만이라 병렬 수집이 동시에 통과하면 deferred 중복 행이 영속될 수 있다.
-- 처방: WHERE status IN ('registered','deferred') 로 재정의(같은 이름 유지 대신 명시적
--   신규 이름 — 의미가 바뀌었음을 인덱스명으로 드러냄). 선행 확인(2026-07-15 dev 실DB):
--   registered/deferred 중복 해시 0건 → CREATE UNIQUE 안전.
-- received/failed 등 비종료 상태·file_hash NULL 은 여전히 제약 대상 외.
-- 운영 주의(migration-reviewer): 트랜잭션 내 DDL 이라 CONCURRENTLY 불가 — 재빌드 동안 asset 락.
--   행수 큰 운영 DB 는 off-peak 배포 창 권고(dev 1148행은 즉시).
-- =============================================================================

DROP INDEX IF EXISTS uq_asset_file_hash_registered;

CREATE UNIQUE INDEX IF NOT EXISTS uq_asset_file_hash_dedup
    ON asset (file_hash)
    WHERE status IN ('registered', 'deferred') AND file_hash IS NOT NULL;

COMMENT ON INDEX uq_asset_file_hash_dedup IS
    'registered/deferred 자산은 동일 file_hash 중복 불가(009 dedup 범위와 정합 — 병렬 수집 동시 통과 안전망·069 B11).';
