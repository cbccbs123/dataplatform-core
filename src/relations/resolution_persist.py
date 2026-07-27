"""관계 생성 재시도 큐 — 관계를 못 만든 자산을 추적한다.

cross-asset 관계 생성(``run_relations``)이 자산별로 관계를 못 만든 경우를 추적하는 경량 큐
``relation_resolution`` 의 결정 로직(순수)과 upsert/조회(conn-우선)를 모은다.

구성
    * ``decide_resolution_status`` — 엣지수·attempts·예외·상한 → (status, attempts) **순수 함수**(DB 불요).
    * ``upsert_resolution``        — 자산당 1행 큐 upsert(``ON CONFLICT (asset_id) DO UPDATE``).
    * ``fetch_unresolved_asset_ids`` — ``--retry`` 대상(pending + 미시도) 자산 id 선택(결정적 정렬).

설계 불변식
    * 큐는 ``asset.status`` FSM·CHECK 제약·``ALLOWED_TRANSITIONS`` 를 건드리지 않는다(헌법 6·7조).
    * ``last_reason`` 은 비식별 사유만(예외 타입·고립 표식). PHI/풀경로 미포함(헌법 10조).
    * 모든 결정은 결정적(헌법 3조) — 같은 입력에 같은 출력.
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection

# ── 큐 상태 상수 ────────────────────────────────────────────────────────────
# 상태 문자열의 정본은 ``domain/status_vocab`` 이고 DB CHECK 도 그 값을 쓴다 — 여기 상수는
# 실사용 값이라 정본과 **반드시 같아야** 한다(테스트가 동기화를 검증한다).
STATUS_PENDING = "pending"
STATUS_RESOLVED = "resolved"
STATUS_ISOLATED = "isolated"  # 평가는 끝났고 관계가 0건 — **실패가 아니라서** 재시도하지 않는다.
STATUS_FAILED = "failed"


def decide_resolution_status(
    edges_upserted: int,
    attempts: int,
    *,
    error: BaseException | None,
    max_attempts: int,
) -> tuple[str, int]:
    """다음 큐 상태를 결정한다 — **순수 함수**(DB·시각·난수 없음).

    Args:
        edges_upserted: 이번 시도에서 실제로 저장된 엣지 수. 0이면 "관계 없음"이다.
        attempts: 지금까지의 누적 시도 횟수(이번 시도 전 값).
        error: 이번 시도에서 난 예외. ``None`` 이면 성공적으로 평가를 마쳤다는 뜻이다 —
            **엣지가 0이어도 실패가 아니다**(고립).
        max_attempts: 재시도 상한. ``attempts+1`` 이 이 값에 닿으면 DLQ(``failed``)로 보낸다.

    Returns:
        ``(status, attempts)``. status 는 ``resolved``/``isolated``/``pending``/``failed`` 중 하나.

    규칙(결정적):
      * error None & 엣지≥1 → ('resolved', attempts)           # 관계 생성 성공, attempts 불변
      * error None & 엣지==0 → ('isolated', attempts)           # 고립 = 평가 완료·관계 0
      * 예외(error 있음)     → attempts+1, 재시도 대상:
          - attempts+1 < max_attempts → ('pending', attempts+1)   # 일시 실패 재시도
          - attempts+1 >= max_attempts → ('failed', attempts+1)   # 상한 도달 DLQ 승격

    035 #2: 고립(엣지0·예외 없음)은 후보 검색(SQL)·LLM(temp=0)이 결정적이라 재시도해도 동일 결과 →
    'isolated'로 즉시 종료(재시도·attempts 증가·DLQ 없음, LLM 낭비 제거). resolved(관계 있음)와 별도
    상태라 #6(증분 재평가)이 WHERE status='isolated'로 재평가 대상을 깔끔히 집는다. 'isolated'는 미해소
    스캔(pending OR 큐행없음)에서 자연 제외돼 재시도되지 않는다. 일시 실패(예외)만 재시도 흐름 유지 —
    다음 주기에 인프라가 살아있으면 성공할 수 있어 의미가 있다.
    """
    if error is None:
        # 엣지가 있으면 해결, 없으면 고립 — 둘 다 정상 종료라 재시도하지 않는다.
        return (STATUS_RESOLVED if edges_upserted >= 1 else STATUS_ISOLATED), attempts
    # 예외(일시 실패)만 재시도 대상.
    next_attempts = attempts + 1
    if next_attempts >= max_attempts:
        return STATUS_FAILED, next_attempts
    return STATUS_PENDING, next_attempts


def upsert_resolution(
    conn: Connection[Any],
    asset_id: str,
    *,
    status: str,
    attempts: int,
    reason: str | None,
) -> None:
    """자산당 1행짜리 재시도 큐를 갱신한다(없으면 삽입).

    **DB에 쓴다**. 호출자의 트랜잭션 안에서 돈다.

    ⚠️ ``updated_at = now()`` 를 명시적으로 SET 한다 — 테이블에 자동갱신 트리거가 없으므로
    충돌 갱신 시 갱신 시각을 직접 기록해야 한다(insert 시는 컬럼 DEFAULT now() 가 채움).

    Args:
        asset_id: 큐 대상 자산(충돌 기준 키).
        status: ``decide_resolution_status`` 가 정한 상태 문자열. 닫힌 어휘이며 DB CHECK 와
            일치해야 한다.
        attempts: 누적 시도 횟수(그대로 덮어쓴다).
        reason: 마지막 사유. **비식별 문자열만** 넘겨야 한다 — 예외 타입·고립 표식 정도이며
            PHI·전체 경로는 금지다(헌법 10조). 이 함수는 받은 값을 검사 없이 그대로 저장하므로
            비식별 책임은 호출자에게 있다. ``None`` 이면 사유 없이 기록한다.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO relation_resolution (asset_id, status, attempts, last_reason)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (asset_id) DO UPDATE SET
                status      = EXCLUDED.status,
                attempts    = EXCLUDED.attempts,
                last_reason = EXCLUDED.last_reason,
                updated_at  = now()
            """,
            (asset_id, status, attempts, reason),
        )


# ── --retry 대상 선택 베이스 쿼리 ────────────────────────────────────────────
# run_relations._fetch_registered_asset_ids 와 동일한 베이스(status='registered' + 임베딩 존재)에
# relation_resolution 를 LEFT JOIN 하여 미해소(pending)와 미시도(큐 행 없음=asset_id IS NULL)만 고른다.
#   * resolved/failed(DLQ) 자산은 (rr.status='pending' OR rr.asset_id IS NULL) 조건으로 자연 제외.
#   * 정렬 기준: 시도 횟수 → 생성 시각 → id.
#     덜 시도한 자산을 먼저, 동률이면 오래된 순, 시각까지 같으면 id 순 — 순서를 완전히 고정한다.
_UNRESOLVED_SQL = """
    SELECT a.asset_id
    FROM asset a
    LEFT JOIN relation_resolution rr ON rr.asset_id = a.asset_id
    WHERE a.status = 'registered'
      AND EXISTS (SELECT 1 FROM asset_embedding e WHERE e.asset_id = a.asset_id)
      AND (rr.status = 'pending' OR rr.asset_id IS NULL)
    ORDER BY rr.attempts ASC NULLS FIRST, a.created_at ASC, a.asset_id ASC
"""


def fetch_unresolved_asset_ids(conn: Connection[Any]) -> list[str]:
    """``--retry`` 대상 자산 id — pending(미해소) + 미시도(큐 행 없음)만, 결정적 정렬.

    조회 전용(쓰기 없음). 트랜잭션 경계는 호출자가 제어한다.
    미시도 자산은 큐 행이 없어 rr.attempts 가 NULL 이므로 ``NULLS FIRST`` 로 가장 먼저 시도한다.

    Returns:
        자산 id 문자열 리스트. 덜 시도한 것 → 오래된 것 → id 순으로 **결정적** 정렬된다.
        ``resolved``·``isolated``·``failed`` 자산은 애초에 뽑히지 않으므로 빈 리스트일 수 있다.
    """
    with conn.cursor() as cur:
        cur.execute(_UNRESOLVED_SQL)
        return [str(r[0]) for r in cur.fetchall()]
