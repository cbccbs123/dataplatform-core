"""074 — verify_top_assets 순수 단위(가짜 judge 주입·LLM/네트워크 0)."""

from __future__ import annotations

import time
import unittest
from typing import Any

from src.search.llm_verify import verify_top_assets


def _b(**buckets: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return dict(buckets)


def _r(rid: str, sim: float, summ: str = "") -> dict[str, Any]:
    return {"id": rid, "similarity": sim, "summary": summ or f"요약 {rid}"}


class TestVerifyTopAssets(unittest.TestCase):
    def test_drops_judged_irrelevant_from_all_buckets(self) -> None:
        # 상위 3(합산 sim) 중 무관 판정 자산을 전 버킷에서 제거 — 같은 자산이 두 버킷에 있어도 동기 삭제.
        buckets = _b(
            text=[_r("a", 0.9), _r("bad", 0.8)],
            video=[_r("bad", 0.8), _r("c", 0.7), _r("d", 0.1)],
        )
        calls: list[str] = []

        def judge(q: str, aid: str, s: str) -> bool:
            calls.append(aid)
            return aid != "bad"

        out, meta = verify_top_assets(buckets, "질의 원문 문장", norm_query="질의", judge_fn=judge, cache={})
        self.assertEqual([r["id"] for r in out["text"]], ["a"])
        self.assertEqual([r["id"] for r in out["video"]], ["c", "d"])  # 4위 d 는 무접촉(FR-002)
        self.assertEqual(sorted(calls), ["a", "bad", "c"])  # 중복 제거 상위 3만 판정
        self.assertEqual((meta["verified"], meta["dropped"], meta["fallback"]), (3, 1, False))

    def test_deadline_exceeded_falls_back_unchanged(self) -> None:
        # 데드라인 초과 → 전량 폴백(원 버킷 그대로·드롭 0·meta.fallback) — 부분 적용 금지(결정성).
        buckets = _b(text=[_r("a", 0.9), _r("b", 0.8)])

        def slow_judge(q: str, aid: str, s: str) -> bool:
            time.sleep(0.2)
            return False  # 완주했다면 드롭됐을 판정 — 폴백이라 반영되면 안 됨

        out, meta = verify_top_assets(
            buckets, "질의", norm_query="질의", judge_fn=slow_judge, cache={}, deadline_s=0.01
        )
        self.assertEqual([r["id"] for r in out["text"]], ["a", "b"])
        self.assertTrue(meta["fallback"])

    def test_fallback_returns_promptly_without_waiting_judges(self) -> None:
        # 리뷰 🔴 회귀: with-블록 종료(wait=True)가 폴백 반환을 실행 중 judge 완주까지 블록하던 버그.
        # 느린 judge(1.0s) 주입·deadline 0.05s → 함수는 judge 완주를 기다리지 않고 즉시 반환해야 한다.
        buckets = _b(text=[_r("a", 0.9)])

        def slow_judge(q: str, aid: str, s: str) -> bool:
            time.sleep(1.0)
            return False

        t0 = time.perf_counter()
        out, meta = verify_top_assets(
            buckets, "질의", norm_query="질의", judge_fn=slow_judge, cache={}, deadline_s=0.05
        )
        elapsed = time.perf_counter() - t0
        self.assertTrue(meta["fallback"])
        self.assertEqual([r["id"] for r in out["text"]], ["a"])
        self.assertLess(elapsed, 0.6)  # judge(1.0s) 완주 대기 없음(여유 마진 포함)

    def test_judge_exception_falls_back_unchanged(self) -> None:
        buckets = _b(text=[_r("a", 0.9)])

        def bad_judge(q: str, aid: str, s: str) -> bool:
            raise RuntimeError("gemma down")

        out, meta = verify_top_assets(buckets, "질의", norm_query="질의", judge_fn=bad_judge, cache={})
        self.assertEqual([r["id"] for r in out["text"]], ["a"])
        self.assertTrue(meta["fallback"])

    def test_cache_hit_skips_judge_and_applies_verdict(self) -> None:
        # 캐시 적중은 judge 미호출(FR-004) — 저장된 무관 판정이 그대로 적용된다.
        buckets = _b(text=[_r("a", 0.9), _r("bad", 0.8)])
        cache = {("질의", "a"): True, ("질의", "bad"): False}
        calls: list[str] = []

        def judge(q: str, aid: str, s: str) -> bool:
            calls.append(aid)
            return True

        out, meta = verify_top_assets(buckets, "원문", norm_query="질의", judge_fn=judge, cache=cache)
        self.assertEqual(calls, [])  # 전부 캐시 → LLM 0회
        self.assertEqual([r["id"] for r in out["text"]], ["a"])
        self.assertEqual((meta["cache_hits"], meta["dropped"]), (2, 1))

    def test_verdicts_cached_after_run_and_capped(self) -> None:
        # 완주 판정은 캐시에 동결되고, 상한 초과 시 오래된 항목부터 제거된다.
        buckets = _b(text=[_r("a", 0.9)])
        cache: dict = {("옛질의", "x"): True}
        verify_top_assets(
            buckets, "원문", norm_query="질의", judge_fn=lambda q, a, s: True, cache=cache, cache_max=1
        )
        self.assertEqual(list(cache), [("질의", "a")])  # 옛 항목 제거·새 판정만

    def test_empty_buckets_passthrough(self) -> None:
        out, meta = verify_top_assets(_b(text=[]), "질의", norm_query="질의", judge_fn=lambda *a: True, cache={})
        self.assertEqual(out, {"text": []})
        self.assertEqual((meta["verified"], meta["fallback"]), (0, False))

    def test_judge_receives_original_query_and_summary(self) -> None:
        # 판정 프롬프트 입력 계약: 사용자 원문 질의 + 자산 summary(FR-002·의도 정보 보존).
        buckets = _b(text=[_r("a", 0.9, summ="씨름의 역사")])
        seen: list[tuple] = []

        def judge(q: str, aid: str, s: str) -> bool:
            seen.append((q, aid, s))
            return True

        verify_top_assets(buckets, "씨름 기술 종류는 어떻게 돼?", norm_query="씨름 기술", judge_fn=judge, cache={})
        self.assertEqual(seen, [("씨름 기술 종류는 어떻게 돼?", "a", "씨름의 역사")])


if __name__ == "__main__":
    unittest.main()
