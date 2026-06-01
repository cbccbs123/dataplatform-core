"""006 검색 KPI 하니스 e2e(T001/T022/T023) — 결정적 합성 코퍼스로 재현율·응답시간 측정.

- T001: 코퍼스 적재 확인(개수 sanity).
- T022: 재현율 recall@N ≥ 60%(국책 KPI). 합성이라 사실상 ~100%, 회귀 가드.
- T023: 응답시간 p95 ≤ 1000ms. 기본 코퍼스 2000, 국책 목표 @10k 는 ``KPI_CORPUS_N=10000`` 로.

검색은 ``_run_hybrid_search(query_vector=e_q)`` 벡터 주입(모델 없음). RUN_DB_E2E + 실 DB 필요.
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

from dotenv import load_dotenv

from src.search import media_search as ms

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"
_N = int(os.getenv("KPI_CORPUS_N", "2000"))  # 배경 자산 수(국책 KPI 목표는 10000)
_N_QUERIES = 10
_RELEVANT = 5


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만(실 PostgreSQL)")
class TestSearchKpiE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(_ENV, override=False)
        from src.config.settings import init_settings

        init_settings("dev")
        from src.database.postgres_util import PostgresUtil
        from tests.fixtures.search.build_corpus import build_corpus, clear_corpus

        cls.db = PostgresUtil()
        cls.db.__enter__()
        clear_corpus(cls.db)  # 잔여 정리 후 깨끗한 적재
        cls.goldens = build_corpus(
            cls.db, n_background=_N, n_queries=_N_QUERIES, relevant_per_query=_RELEVANT
        )
        cls.total = _N + _N_QUERIES * _RELEVANT

    @classmethod
    def tearDownClass(cls) -> None:
        from tests.fixtures.search.build_corpus import clear_corpus

        clear_corpus(cls.db)
        cls.db.__exit__(None, None, None)

    def _search(self, golden: dict, *, limit: int = 20) -> list[dict]:
        return ms._run_hybrid_search(
            query_vector=golden["query_vec"],
            bm25_query=golden["query_text"],
            media_types=["txt"],
            embedding_kind="st",
            limit=limit,
            alpha=0.75,
        )

    def test_corpus_loaded(self) -> None:  # T001
        from tests.fixtures.search.build_corpus import count_corpus

        self.assertEqual(count_corpus(self.db), self.total)

    def test_recall_at_n_ge_60pct(self) -> None:  # T022
        recalls = []
        for g in self.goldens:
            top_ids = {r["id"] for r in self._search(g, limit=20)[:20]}
            found = len(top_ids & g["relevant_ids"])
            recalls.append(found / len(g["relevant_ids"]))
        mean_recall = sum(recalls) / len(recalls)
        print(f"[KPI] recall@20 평균={mean_recall:.3f} (코퍼스 {self.total}건, 질의 {len(self.goldens)})")
        self.assertGreaterEqual(mean_recall, 0.60)  # 국책 KPI: 재현율 ≥ 60%

    def test_latency_p95_le_1000ms(self) -> None:  # T023
        # 워밍업 1회(인덱스/플랜 캐시) 후 측정
        self._search(self.goldens[0])
        lat_ms: list[float] = []
        runs = 30
        for i in range(runs):
            g = self.goldens[i % len(self.goldens)]
            t0 = time.perf_counter()
            self._search(g)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
        lat_ms.sort()
        p95 = lat_ms[min(len(lat_ms) - 1, int(0.95 * len(lat_ms)))]
        print(
            f"[KPI] p95={p95:.1f}ms (코퍼스 {self.total}건, {runs}회). "
            f"국책 목표 @10k — 현재 N={_N} (KPI_CORPUS_N 로 조정)"
        )
        self.assertLessEqual(p95, 1000.0)  # 국책 KPI: 응답 ≤ 1000ms


if __name__ == "__main__":
    unittest.main()
