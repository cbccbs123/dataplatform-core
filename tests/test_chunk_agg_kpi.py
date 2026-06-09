"""청크 집계 방식 KPI 측정 하니스 (019 G3 — T007·T009 순수 계산부).

017 A/B KPI 하니스(``tests/test_embedding_ab_kpi.py``)의 **축**을 채널/모델에서 **집계 방식**
(``agg ∈ {max, topk_mean(k=3), mix}``)으로 바꾼 동형 측정부를 검증한다. 017 이 같은 골든셋을
두 채널로 돌려 recall@20/MRR/nDCG 를 산출했듯, 019 는 같은 골든셋·같은 채널을 **집계 방식별**로
돌려 같은 지표 + 무관 노이즈 지표를 산출한다.

테스트 전략(docs/테스트_가이드.md §0·§2)
  - **순수 계산부(DB 무관)**: 실제 검색 실행은 **주입 seam**(``search_fn(query, payload)``)으로
    대체해 가짜 랭킹을 주입하고, 메트릭 집계 로직(``compute_agg_kpi_table``)만 단위 검증한다.
    실DB 검색 호출(``search_hybrid``)은 G4(사람, ``RUN_DB_E2E``)에서 돈다 — 여기서는 검색 결과를
    주입하므로 DB·모델이 전혀 필요 없다.
  - **러너 조립부**: ``make_db_search_fn`` 이 ``search_hybrid`` 호출에 ``chunk_agg`` 를 정확히
    실어 보내는지(축 주입), ``build_agg_variants`` 가 측정 축 3종을 내는지 가짜로 검증한다.

측정 러너 모듈(``scripts/measure_chunk_agg_kpi.py``)을 importlib 로 적재한다(scripts 는 패키지가
아니라, 017 의 ``build_golden_ko_draft`` 적재 방식과 동형). 러너의 **계산·조립 로직**은 본 파일이
단위로 덮고, **실DB 검색 실행 자체**는 러너 ``main()``(RUN_DB_E2E 게이트) 에서 G4 가 돌린다.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[1]
_RUNNER_PATH = _REPO / "scripts" / "measure_chunk_agg_kpi.py"


def _load_runner() -> ModuleType:
    """``scripts/measure_chunk_agg_kpi.py`` 를 모듈로 적재(017 build_golden_ko_draft 적재와 동형)."""
    spec = importlib.util.spec_from_file_location("measure_chunk_agg_kpi", _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 손계산 가능한 결정적 가짜 랭킹(이미 평가 풀 한정·합산된 id 순위). 집계 방식 축이 다르면
# 랭킹이 달라져 지표가 갈리는 것을 단위로 못박는다. q3 정답은 풀 밖이라 평가 제외(017 동형).
_FAKE_RANKINGS = {
    ("q1", "max"): ["d", "a", "e", "b"],       # a@2, b@4
    ("q2", "max"): ["e", "f", "c"],            # c@3
    ("q1", "topk_mean"): ["a", "b", "d", "e"], # a@1, b@2 (무관 d·e 가 아래로)
    ("q2", "topk_mean"): ["c", "e", "f"],      # c@1
}

_GOLDENS = [
    {"query": "q1", "relevant_asset_ids": ["a", "b"]},
    {"query": "q2", "relevant_asset_ids": ["c"]},
    {"query": "q3", "relevant_asset_ids": ["zzz"]},  # 풀 밖 정답 → 평가 제외(017 _compute 동형)
]
_POOL = {"a", "b", "c", "d", "e", "f"}
_VARIANTS = [("max", "max"), ("topk_mean", "topk_mean")]  # (라벨, payload). payload 는 seam 이 해석.


def _fake_search(query: str, payload):
    """주입 seam: (질의, 집계 payload) → 결정적 가짜 랭킹. 실DB·모델 없이 메트릭 로직만 검증."""
    return list(_FAKE_RANKINGS[(query, payload)])


class TestComputeAggKpiTable(unittest.TestCase):
    """집계 방식 축별 KPI 집계(순수 계산부, 주입 seam) — 017 _compute 의 집계-축 버전."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_runner()

    def _table(self):
        return self.mod.compute_agg_kpi_table(
            _GOLDENS, _POOL, _VARIANTS, _fake_search, k=20, expose_ns=(5, 10)
        )

    def test_skips_query_with_no_relevant_in_pool(self) -> None:
        # q3 정답(zzz)은 풀 밖 → 평가 질의 2건(q1·q2)만 집계(017 과 동일 규칙).
        t = self._table()
        self.assertEqual(t["evaluated"], 2)
        self.assertEqual(t["pool_size"], len(_POOL))
        self.assertEqual(t["variants"], ["max", "topk_mean"])

    def test_relevance_metrics_per_agg_variant(self) -> None:
        # 같은 골든셋·풀에서 집계 방식만 바꿔 recall@20/MRR/nDCG@20 을 산출(손계산 대조).
        from tests.fixtures.search.metrics import ndcg_at_k

        m = self._table()["metrics"]
        # max: q1 MRR=1/2, q2 MRR=1/3 → 평균 0.41667; recall 둘 다 1.0.
        self.assertAlmostEqual(m["max"]["recall@20"], 1.0, places=6)
        self.assertAlmostEqual(m["max"]["MRR"], (0.5 + 1 / 3) / 2, places=6)
        # nDCG 는 매직 상수 대신 검증된 primitive 로 오라클을 재유도해 집계 평균을 못박는다.
        ndcg_max = (
            ndcg_at_k(["d", "a", "e", "b"], {"a", "b"}, 20)
            + ndcg_at_k(["e", "f", "c"], {"c"}, 20)
        ) / 2
        self.assertAlmostEqual(m["max"]["nDCG@20"], ndcg_max, places=9)
        self.assertGreater(ndcg_max, 0.5)  # 집계가 실제로 nDCG 를 산출함(0 아님)
        # topk_mean: 정답이 최상위 → MRR·nDCG 모두 1.0(집계 개선이 지표로 드러남)
        self.assertAlmostEqual(m["topk_mean"]["recall@20"], 1.0, places=6)
        self.assertAlmostEqual(m["topk_mean"]["MRR"], 1.0, places=6)
        self.assertAlmostEqual(m["topk_mean"]["nDCG@20"], 1.0, places=6)

    def test_noise_metrics_per_agg_variant(self) -> None:
        # 무관 노이즈 지표: 정답 아닌 자산의 평균 순위(높을수록 좋음)·상위-N 노출률(낮을수록 좋음).
        m = self._table()["metrics"]
        # max: 무관 평균순위 q1=(1+3)/2=2.0, q2=(1+2)/2=1.5 → 평균 1.75
        self.assertAlmostEqual(m["max"]["noise_mean_rank"], (2.0 + 1.5) / 2, places=6)
        # topk_mean: 무관이 아래로 밀려 평균순위↑ q1=(3+4)/2=3.5, q2=(2+3)/2=2.5 → 3.0
        self.assertAlmostEqual(m["topk_mean"]["noise_mean_rank"], (3.5 + 2.5) / 2, places=6)
        # 상위-5 노출률: q1=2/4=0.5, q2=2/3 → 평균 (0.5+0.66667)/2
        self.assertAlmostEqual(m["max"]["noise@5"], (0.5 + 2 / 3) / 2, places=6)
        self.assertAlmostEqual(m["topk_mean"]["noise@5"], (0.5 + 2 / 3) / 2, places=6)
        # noise@10 은 랭킹 길이<10 이라 noise@5 와 동일(top 슬롯 동일)
        self.assertAlmostEqual(m["max"]["noise@10"], m["max"]["noise@5"], places=6)

    def test_deterministic_two_calls_equal(self) -> None:
        # 같은 입력·같은 seam → 2회 동일(헌법 3조). per_query 까지 포함해 동일해야 함.
        self.assertEqual(self._table(), self._table())

    def test_empty_relevant_pool_intersection_raises_or_zero_eval(self) -> None:
        # 모든 골든 정답이 풀 밖이면 평가 질의 0건 — evaluated=0 으로 정직하게 드러낸다(017 동형).
        t = self.mod.compute_agg_kpi_table(
            [{"query": "q1", "relevant_asset_ids": ["zzz"]}],
            _POOL,
            _VARIANTS,
            _fake_search,
            k=20,
            expose_ns=(5, 10),
        )
        self.assertEqual(t["evaluated"], 0)


class TestFormatComparisonTable(unittest.TestCase):
    """비교표 포매팅(순수) — 집계 방식별 지표 + 노이즈 지표를 사람이 읽는 표로."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_runner()

    def test_table_contains_variants_and_metrics(self) -> None:
        table = self.mod.compute_agg_kpi_table(
            _GOLDENS, _POOL, _VARIANTS, _fake_search, k=20, expose_ns=(5, 10)
        )
        text = self.mod.format_comparison_table(table)
        self.assertIsInstance(text, str)
        # 축(집계 방식) 라벨과 지표명이 표에 노출돼야 한다(측정 결과를 사람이 읽음).
        for label in ("max", "topk_mean"):
            self.assertIn(label, text)
        for metric in ("recall@20", "MRR", "nDCG@20", "noise_mean_rank", "noise@5"):
            self.assertIn(metric, text)
        self.assertIn("평가 질의 2", text)  # evaluated 노출


class TestRunnerAssembly(unittest.TestCase):
    """러너 조립부(축 주입·변형 목록) — 실DB 없이 가짜로 검증(실DB 호출은 G4)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_runner()

    def test_build_agg_variants_axis(self) -> None:
        # 측정 축 3종: max / topk_mean(k=3) / mix(w=0.5). 017 채널 축의 집계 버전.
        from src.config.settings import ChunkAggConfig

        variants = self.mod.build_agg_variants()
        labels = [label for label, _ in variants]
        self.assertEqual(labels, ["max", "topk_mean", "mix"])
        for label, payload in variants:
            self.assertIsInstance(payload, ChunkAggConfig)
            self.assertEqual(payload.agg, label)
            self.assertEqual(payload.k, 3)
            self.assertEqual(payload.mix_w, 0.5)

    def test_make_db_search_fn_threads_chunk_agg(self) -> None:
        # seam 조립: search_hybrid 에 chunk_agg(축 payload)·멀티모달을 정확히 실어 보내고,
        # 017 _merge_ranked_ids(주입) 결과를 그대로 반환한다. 실DB 호출은 fake 로 대체.
        captured: dict = {}

        def fake_hybrid(query, **kw):
            captured["query"] = query
            captured.update(kw)
            return {"results": {"text_documents": [{"id": "x", "similarity": 0.9}]}}

        def fake_merge(results, pool):
            captured["merge_results"] = results
            captured["merge_pool"] = pool
            return ["x", "y"]

        fn = self.mod.make_db_search_fn(
            {"x", "y"}, search_hybrid_fn=fake_hybrid, merge_fn=fake_merge
        )
        out = fn("스마트폰 생산량", "AGG_SENTINEL")
        self.assertEqual(out, ["x", "y"])
        self.assertEqual(captured["chunk_agg"], "AGG_SENTINEL")
        self.assertEqual(captured["modalities"], ["text", "audio", "video"])
        self.assertEqual(captured["merge_pool"], {"x", "y"})


if __name__ == "__main__":
    unittest.main()
