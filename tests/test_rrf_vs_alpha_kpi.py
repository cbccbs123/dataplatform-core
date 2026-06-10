"""alpha 가중합 vs RRF 융합 KPI 비교(신호 충돌 합성 코퍼스). RUN_DB_E2E=1 + 실 DB 필요.

같은 후보 풀(alpha-cap 전, 큰 limit)에서 emb·bm25 독립 랭킹을 RRF로 합쳐
alpha(similarity 순)와 recall@5/precision@5/MRR/nDCG@5 를 비교한다.
핵심 단언: lexical-강 정답(L) 재현율에서 RRF ≥ alpha(0.75=임베딩 편향)(설계 §9-2).

벡터 직접 주입(모델 없음) → 결정적(헌법 3조). /kpi_corpus_rrf/ 마커 자산만 생성·삭제.
"""

from __future__ import annotations

import os
import statistics as st
import unittest
from pathlib import Path

from dotenv import load_dotenv

from src.search import media_search as ms
from tests.fixtures.search.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"
_METRICS = ("R@5", "P@5", "MRR", "nDCG@5", "L_recall")


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만(실 PostgreSQL)")
class TestRRFvsAlpha(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(_ENV, override=False)
        from src.config.settings import init_settings

        init_settings("dev")
        from src.database.postgres_util import PostgresUtil
        from tests.fixtures.search.build_corpus import (
            build_disagreement_corpus,
            clear_disagreement_corpus,
        )

        cls.db = PostgresUtil()
        cls.db.__enter__()
        clear_disagreement_corpus(cls.db)  # 잔여 정리 후 깨끗한 적재
        cls.goldens = build_disagreement_corpus(cls.db)

    @classmethod
    def tearDownClass(cls) -> None:
        from tests.fixtures.search.build_corpus import clear_disagreement_corpus

        clear_disagreement_corpus(cls.db)
        cls.db.__exit__(None, None, None)

    def _ranked(self, g: dict, fusion: str) -> list[str]:
        # alpha-cap 전의 넉넉한 풀(limit=500)에서 두 방식이 같은 후보를 보게 한다(공정성, 설계 §2.2).
        rows = ms._run_hybrid_search(
            query_vector=g["query_vec"],
            bm25_query=g["query_text"],
            media_types=["txt"],
            embedding_kind="st",
            limit=500,
            alpha=0.75,
            fusion=fusion,
        )
        return [str(r["id"]) for r in rows]

    def test_compare(self) -> None:
        agg = {f: {m: [] for m in _METRICS} for f in ("alpha", "rrf")}
        for g in self.goldens:
            rel, lset = g["relevant_ids"], g["classes"]["L"]
            for fusion in ("alpha", "rrf"):
                rk = self._ranked(g, fusion)
                agg[fusion]["R@5"].append(recall_at_k(rk, rel, 5))
                agg[fusion]["P@5"].append(precision_at_k(rk, rel, 5))
                agg[fusion]["MRR"].append(mrr(rk, rel))
                agg[fusion]["nDCG@5"].append(ndcg_at_k(rk, rel, 5))
                agg[fusion]["L_recall"].append(recall_at_k(rk, lset, 5))

        print(f"\n[RRF vs alpha KPI] (충돌 코퍼스, 질의 {len(self.goldens)})")
        for m in _METRICS:
            a, r = st.mean(agg["alpha"][m]), st.mean(agg["rrf"][m])
            print(f"  {m:<9} alpha={a:.3f}  rrf={r:.3f}  Δ={r - a:+.3f}")

        # 핵심 단언: lexical-강 정답(L) 재현율에서 RRF가 alpha(임베딩 편향)보다 우위(설계 §9-2).
        self.assertGreaterEqual(
            st.mean(agg["rrf"]["L_recall"]), st.mean(agg["alpha"]["L_recall"])
        )


if __name__ == "__main__":
    unittest.main()
