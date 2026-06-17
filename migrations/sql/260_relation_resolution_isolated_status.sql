-- 260 — relation_resolution.status 에 'isolated' 추가 (spec 035 #2: 고립 ≠ 실패).
--
-- 배경: 종전 큐는 고립(후보0/엣지0·예외 없음)과 일시 실패(예외)를 모두 pending→상한 시 failed(DLQ)로
--   동일 취급해, 고립 자산이 LLM 을 max_attempts 회 낭비 호출 후 DLQ 에 영구 격리됐다. 고립은 정상
--   평가 결과(파트너 없음)이지 실패가 아니다.
-- 변경:
--   (1) status CHECK 어휘에 'isolated' 추가 — resolved(관계≥1)·isolated(관계0·평가완료)·failed(오류 DLQ)·pending 구분.
--       decide_resolution_status 가 고립을 'isolated'로 전이(재시도·attempts·DLQ 없음). 향후 #6(증분 재평가)이
--       WHERE status='isolated' 로 재평가 대상을 깔끔히 집는다.
--   (2) backfill — 과거 고립이 DLQ(failed)로 잘못 격리된 행을 'isolated'로 되살린다. 식별 키 = failed +
--       last_reason='isolated:no_edges'(035 전 호출부가 고립에 단 표식; 예외 실패는 예외 타입명이라 제외됨). 멱등.
--   ※ 부분 인덱스(WHERE status='pending')·스캔(pending OR 큐행없음)·asset 계층 무손상 — 'isolated'는 미해소 스캔에서
--     자연 제외(pending 아님)되어 재시도되지 않는다.

ALTER TABLE relation_resolution DROP CONSTRAINT IF EXISTS relation_resolution_status_check;
ALTER TABLE relation_resolution ADD CONSTRAINT relation_resolution_status_check
    CHECK (status IN ('pending', 'resolved', 'isolated', 'failed'));

-- DLQ 에 갇힌 고립 행 복구(멱등) — 예외 실패(last_reason=예외 타입명)는 건드리지 않는다.
-- updated_at 도 now() 로 갱신: 자동 트리거가 없어(v250) 상태 전이 시각을 명시해야 #6 증분 재평가가
-- updated_at 을 staleness 신호로 쓸 때 일관적이다(migration-reviewer 권고).
UPDATE relation_resolution
   SET status = 'isolated', updated_at = now()
 WHERE status = 'failed' AND last_reason = 'isolated:no_edges';

COMMENT ON COLUMN relation_resolution.status IS
    '큐 상태: pending(미해소·일시 실패 재시도 대기) | resolved(관계≥1 생성) | isolated(평가 완료·관계 0, #6 재평가 대상) | failed(재시도 상한 도달 DLQ).';
