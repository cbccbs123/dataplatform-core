"""073 — aboutness 백필 코어(seam 주입·실 DB/LLM 0) 격리·카운트 검증."""

from __future__ import annotations

import unittest

from src.app.run_about_backfill import backfill_about


def _targets(*pairs: tuple[str, str | None]) -> list[dict]:
    return [{"asset_id": a, "summary": s} for a, s in pairs]


class TestBackfillAbout(unittest.TestCase):
    def test_counts_done_and_empty(self) -> None:
        # 추출 성공(비어있는 about 포함)을 done 으로, 빈 about 은 empty 로 함께 센다.
        def extract(aid, summary):
            return ["개체"] if summary else []

        counts = backfill_about(
            _targets(("a1", "요약"), ("a2", None)), extract_persist_fn=extract
        )
        self.assertEqual(counts, {"done": 2, "empty": 1, "failed": 0, "os_failed": 0})

    def test_per_asset_isolation_on_failure(self) -> None:
        # 한 자산의 예외는 failed 카운트 후 계속(배치 격리) — OS 갱신도 실패 자산은 건너뜀.
        os_calls: list[str] = []

        def extract(aid, summary):
            if aid == "bad":
                raise RuntimeError("LLM down")
            return ["x"]

        counts = backfill_about(
            _targets(("a1", "s"), ("bad", "s"), ("a3", "s")),
            extract_persist_fn=extract,
            os_update_fn=lambda aid, about: os_calls.append(aid),
        )
        self.assertEqual(counts["done"], 2)
        self.assertEqual(counts["failed"], 1)
        self.assertEqual(os_calls, ["a1", "a3"])  # 실패 자산 OS 미호출

    def test_os_failure_swallowed(self) -> None:
        # OS 갱신 실패는 정본(DB 커밋)과 무관 — os_failed 로만 세고 done 유지·배치 계속.
        def os_update(aid, about):
            raise RuntimeError("OS down")

        counts = backfill_about(
            _targets(("a1", "s")), extract_persist_fn=lambda a, s: ["x"], os_update_fn=os_update
        )
        self.assertEqual(counts, {"done": 1, "empty": 0, "failed": 0, "os_failed": 1})


if __name__ == "__main__":
    unittest.main()
