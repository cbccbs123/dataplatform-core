-- =============================================================================
-- F-5.1 보강 — asset.status 에 'deferred' 추가
-- =============================================================================
-- 의료 표준 포맷(DICOM/HL7/FHIR)은 medical 로 분류·등록하되, 추출은 의료 어댑터(후속)에서
-- 진행한다. 그 사이 상태를 'deferred'(추출 보류, 실패 아님)로 둔다. status='deferred' 로 조회.
-- =============================================================================

ALTER TABLE asset DROP CONSTRAINT IF EXISTS asset_status_check;
ALTER TABLE asset ADD CONSTRAINT asset_status_check
    CHECK (status IN ('received', 'routing', 'classifying', 'extracting', 'registered', 'failed', 'deferred'));

COMMENT ON COLUMN asset.status IS
    '처리 상태머신: received→routing→classifying→extracting→registered, 실패 시 failed, 의료 표준 포맷 추출 보류 시 deferred.';
