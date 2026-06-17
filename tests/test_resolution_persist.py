"""009 US3 — decide_resolution_status 순수 단위 테스트.

관계 재시도/미해소 큐의 다음 상태를 결정하는 순수 함수(`decide_resolution_status`)를
DB 없이 검증한다. 엣지수·attempts·예외·상한의 결정적 함수다(헌법 3조 결정 재현성).

규칙(FR-008 + 035 #2):
  * error None & 엣지≥1 → ('resolved', attempts)          ※ 성공(관계 있음·attempts 불변)
  * error None & 엣지==0 → ('isolated', attempts)          ※ 035 #2 고립 = 평가 완료·관계 0(재시도·DLQ 없음, #6 재평가 대상)
  * 예외(error 있음)     → ('pending', attempts+1)          ※ 일시 실패만 재시도
    단, attempts+1 이 max_attempts 에 도달 시 → ('failed', max_attempts)  ※ DLQ 승격
"""

from __future__ import annotations

import unittest

from src.relations.resolution_persist import decide_resolution_status


class TestDecideResolutionStatus(unittest.TestCase):
    # ── 엣지≥1 → resolved (attempts 불변) ────────────────────────────────────
    def test_edges_one_resolved(self) -> None:
        status, attempts = decide_resolution_status(1, 0, error=None, max_attempts=3)
        self.assertEqual(status, "resolved")
        self.assertEqual(attempts, 0)

    def test_edges_many_resolved_attempts_unchanged(self) -> None:
        # 이미 2회 실패했어도 이번에 엣지가 생기면 resolved, attempts 는 그대로.
        status, attempts = decide_resolution_status(5, 2, error=None, max_attempts=3)
        self.assertEqual(status, "resolved")
        self.assertEqual(attempts, 2)

    # ── 엣지0 & error None → isolated (035 #2 고립) ──────────────────────────
    def test_edges_zero_isolation_isolated(self) -> None:
        # 035 #2: 고립(후보0/관계0·예외 없음)은 평가 완료 → isolated, attempts 불변(재시도 없음).
        status, attempts = decide_resolution_status(0, 0, error=None, max_attempts=3)
        self.assertEqual(status, "isolated")
        self.assertEqual(attempts, 0)

    def test_edges_zero_isolation_never_escalates_to_failed(self) -> None:
        # 고립은 attempts 가 상한 근처여도 failed(DLQ)로 가지 않는다 — isolated 즉시 종료.
        status, attempts = decide_resolution_status(0, 2, error=None, max_attempts=3)
        self.assertEqual(status, "isolated")
        self.assertEqual(attempts, 2)

    # ── 예외 → pending(attempts+1) 일시 실패 ─────────────────────────────────
    def test_error_pending_increment(self) -> None:
        status, attempts = decide_resolution_status(0, 0, error=RuntimeError("boom"), max_attempts=3)
        self.assertEqual(status, "pending")
        self.assertEqual(attempts, 1)

    def test_error_overrides_positive_edges(self) -> None:
        # 예외가 있으면 edges_upserted 값과 무관하게 실패로 본다(부분 적재 후 예외 등).
        status, attempts = decide_resolution_status(3, 0, error=ValueError("x"), max_attempts=3)
        self.assertEqual(status, "pending")
        self.assertEqual(attempts, 1)

    # ── 경계값: attempts=max-1 → pending / attempts=max → failed ──────────────
    def test_boundary_attempts_max_minus_one_stays_pending(self) -> None:
        # attempts=2, +1=3 == max → 상한 도달이므로 failed 로 승격된다.
        # (max-1 에서 한 번 더 실패하면 max 에 도달 → DLQ)
        status, attempts = decide_resolution_status(0, 2, error=RuntimeError("e"), max_attempts=3)
        self.assertEqual(status, "failed")
        self.assertEqual(attempts, 3)

    def test_boundary_below_threshold_pending(self) -> None:
        # attempts=1, +1=2 < max(3) → 아직 pending.
        status, attempts = decide_resolution_status(0, 1, error=RuntimeError("e"), max_attempts=3)
        self.assertEqual(status, "pending")
        self.assertEqual(attempts, 2)

    def test_boundary_at_max_failed(self) -> None:
        # attempts 가 max-1(=2)에서 다시 일시 실패 → attempts=3=max → failed(DLQ).
        # 035 #2: failed(DLQ)는 이제 예외(일시 실패)로만 도달한다(고립은 resolved).
        status, attempts = decide_resolution_status(0, 2, error=RuntimeError("e"), max_attempts=3)
        self.assertEqual(status, "failed")
        self.assertEqual(attempts, 3)

    def test_boundary_beyond_max_failed(self) -> None:
        # 방어적: 이미 상한을 넘긴 자산이 또 실패해도 failed 로 고정.
        status, attempts = decide_resolution_status(0, 3, error=RuntimeError("e"), max_attempts=3)
        self.assertEqual(status, "failed")
        self.assertEqual(attempts, 4)

    # ── 다양한 max_attempts ──────────────────────────────────────────────────
    def test_max_attempts_one_immediate_failed(self) -> None:
        # max_attempts=1 이면 첫 일시 실패(attempts 0→1)가 곧 상한 도달 → failed.
        # 035 #2: 예외(일시 실패) 기준 — 고립(error None)은 max_attempts 무관하게 resolved.
        status, attempts = decide_resolution_status(0, 0, error=RuntimeError("e"), max_attempts=1)
        self.assertEqual(status, "failed")
        self.assertEqual(attempts, 1)

    def test_resolved_ignores_max(self) -> None:
        # 상한을 넘긴 attempts 라도 엣지가 생기면 resolved(상한은 실패 누적에만 적용).
        status, attempts = decide_resolution_status(2, 9, error=None, max_attempts=3)
        self.assertEqual(status, "resolved")
        self.assertEqual(attempts, 9)


if __name__ == "__main__":
    unittest.main()
