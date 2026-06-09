"""019 G2 — per-asset 청크 집계식 빌더(``_chunk_agg_expr``)·SQL 주입 순수 단위 테스트.

검색 시점 자산 점수의 청크 집계를 ``MAX`` 고정에서 설정 선택식(``max``|``topk_mean``|``mix``)으로
바꾼다(spec 019). 빌더와 3개 집계 지점(ST 하이브리드 ``emb_score``·시각 2단계 ``s_text``/``s_clip``)의
SQL 생성을 **DB 없이 문자열로** 검증한다(실 DB 동작은 G4/사람 몫).

🔴 최우선 안전 게이트(SC-001 회귀 0):
- ``agg='max'`` 의 전체 SQL 은 종전과 **바이트 동일**해야 한다. 본 테스트는 현 빌더 출력의 스냅샷을
  픽스처(``tests/fixtures/chunk_agg/*_max.sql`` — 변경 전 함수 출력 그대로 캡처)로 고정하고,
  ``chunk_agg=max`` 및 기본(미지정) 출력이 그 골든과 문자 동일함을 단언한다(미래 드리프트 차단).
- 결정성(헌법 3조): topk_mean 의 상위 k 선택 윈도우는 ``ORDER BY <sim> DESC NULLS LAST, chunk_index ASC``
  안정 정렬 키를 포함해 동점 청크에서 흔들리지 않아야 한다. 같은 인자 2회 → 동일 문자열.
- 산식만 바뀌고 **바인딩 파라미터(%s) 개수·순서는 불변** — max/topk_mean/mix 의 ``%s`` 개수가
  같아야 호출부 execute 튜플이 그대로다(검증 포함).
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.config.settings import ChunkAggConfig
from src.search.media_search import (
    _chunk_agg_expr,
    _hybrid_embedding_bm25_sql,
    _two_stage_clip_for_ids_sql,
    _two_stage_stage1_sql,
)

_FIX = Path(__file__).resolve().parent / "fixtures" / "chunk_agg"

_MAX = ChunkAggConfig(agg="max", k=3, mix_w=0.5)
_TOPK = ChunkAggConfig(agg="topk_mean", k=3, mix_w=0.5)
_MIX = ChunkAggConfig(agg="mix", k=3, mix_w=0.5)


def _golden(name: str) -> str:
    return (_FIX / name).read_text()


class TestChunkAggExprBuilder(unittest.TestCase):
    """순수 빌더 ``_chunk_agg_expr(agg, *, sim_expr, k, w)`` 의 식 생성."""

    def test_max_is_plain_max_of_sim_expr(self) -> None:
        # max → 종전 집계식과 문자 동일(MAX(<sim_expr>)). 회귀 0 의 식 단위 증명.
        self.assertEqual(
            _chunk_agg_expr("max", sim_expr="chunk_sim", k=3, w=0.5),
            "MAX(chunk_sim)",
        )
        self.assertEqual(
            _chunk_agg_expr(
                "max",
                sim_expr="CASE WHEN chunk_sim = chunk_sim THEN chunk_sim END",
                k=3,
                w=0.5,
            ),
            "MAX(CASE WHEN chunk_sim = chunk_sim THEN chunk_sim END)",
        )

    def test_topk_mean_uses_rank_window_and_k(self) -> None:
        # 상위 k 평균: ranked CTE 의 순위 컬럼(agg_rn) <= k 청크만 AVG. k 가 식에 반영.
        self.assertEqual(
            _chunk_agg_expr("topk_mean", sim_expr="agg_sim", k=3, w=0.5),
            "AVG(CASE WHEN agg_rn <= 3 THEN agg_sim END)",
        )
        self.assertEqual(
            _chunk_agg_expr("topk_mean", sim_expr="agg_sim", k=5, w=0.5),
            "AVG(CASE WHEN agg_rn <= 5 THEN agg_sim END)",
        )

    def test_mix_is_weighted_max_avg(self) -> None:
        # mix → w*MAX + (1-w)*AVG 구조, w 반영.
        self.assertEqual(
            _chunk_agg_expr("mix", sim_expr="agg_sim", k=3, w=0.5),
            "(0.5 * MAX(agg_sim) + (1 - 0.5) * AVG(agg_sim))",
        )
        self.assertEqual(
            _chunk_agg_expr("mix", sim_expr="agg_sim", k=3, w=0.7),
            "(0.7 * MAX(agg_sim) + (1 - 0.7) * AVG(agg_sim))",
        )

    def test_unsupported_agg_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            _chunk_agg_expr("median", sim_expr="agg_sim", k=3, w=0.5)

    def test_deterministic_same_args_same_string(self) -> None:
        # 같은 인자 2회 → 동일 문자열(결정성, 헌법 3조).
        for agg in ("max", "topk_mean", "mix"):
            a = _chunk_agg_expr(agg, sim_expr="agg_sim", k=3, w=0.5)
            b = _chunk_agg_expr(agg, sim_expr="agg_sim", k=3, w=0.5)
            self.assertEqual(a, b, f"{agg} 빌더가 비결정적")


class TestMaxSqlByteIdentical(unittest.TestCase):
    """🔴 SC-001: agg='max' 전체 SQL 이 변경 전 스냅샷과 **바이트 동일**(회귀 0)."""

    def test_hybrid_max_matches_snapshot(self) -> None:
        self.assertEqual(
            _hybrid_embedding_bm25_sql(1536, chunk_agg=_MAX), _golden("hybrid_max.sql")
        )

    def test_hybrid_default_is_max_snapshot(self) -> None:
        # 미지정(기본) = max 바이트 동일 — 기존 bare 호출(006/017 단위)이 그대로 동작.
        self.assertEqual(_hybrid_embedding_bm25_sql(1536), _golden("hybrid_max.sql"))

    def test_stage1_max_matches_snapshot(self) -> None:
        self.assertEqual(
            _two_stage_stage1_sql(1536, chunk_agg=_MAX), _golden("stage1_max.sql")
        )
        self.assertEqual(_two_stage_stage1_sql(1536), _golden("stage1_max.sql"))

    def test_clip_max_matches_snapshot(self) -> None:
        self.assertEqual(
            _two_stage_clip_for_ids_sql(1536, chunk_agg=_MAX), _golden("clip_max.sql")
        )
        self.assertEqual(_two_stage_clip_for_ids_sql(1536), _golden("clip_max.sql"))


class TestTopkMeanSqlStructure(unittest.TestCase):
    """topk_mean: 윈도우 ranked CTE + 안정 정렬 키 + 상위 k 평균. tail(랭킹)은 불변."""

    def test_hybrid_topk_mean_has_stable_window_and_avg(self) -> None:
        sql = _hybrid_embedding_bm25_sql(1536, chunk_agg=_TOPK)
        self.assertIn("emb_ranked AS (", sql)
        self.assertIn("row_number() OVER (", sql)
        self.assertIn("PARTITION BY id", sql)
        # 안정 정렬 키: sim DESC NULLS LAST 후 chunk_index(=chunk_pos) ASC 2차 정렬(결정성).
        self.assertIn("DESC NULLS LAST, chunk_pos ASC", sql)
        self.assertIn("AVG(CASE WHEN agg_rn <= 3 THEN agg_sim END) AS emb_score", sql)
        # NaN 가드 보존(chunk_sim = chunk_sim).
        self.assertIn("CASE WHEN chunk_sim = chunk_sim THEN chunk_sim END", sql)
        # tail(랭킹·정렬)은 max 와 동일하게 유지.
        self.assertIn("ORDER BY similarity DESC NULLS LAST, id ASC", sql)

    def test_hybrid_topk_mean_same_param_count_as_max(self) -> None:
        # 산식만 바뀌고 바인딩 %s 개수는 불변 → execute 튜플 그대로(안전 게이트).
        max_sql = _hybrid_embedding_bm25_sql(1536, chunk_agg=_MAX)
        topk_sql = _hybrid_embedding_bm25_sql(1536, chunk_agg=_TOPK)
        self.assertEqual(topk_sql.count("%s"), max_sql.count("%s"))

    def test_stage1_topk_mean_structure(self) -> None:
        sql = _two_stage_stage1_sql(1536, chunk_agg=_TOPK)
        self.assertIn("stage1_ranked AS (", sql)
        self.assertIn("row_number() OVER (", sql)
        self.assertIn("DESC NULLS LAST, chunk_pos ASC", sql)
        self.assertIn("AVG(CASE WHEN agg_rn <= 3 THEN agg_sim END) AS s_text", sql)
        self.assertIn("ORDER BY s_text DESC, id ASC", sql)  # 정렬 불변
        self.assertEqual(
            sql.count("%s"), _two_stage_stage1_sql(1536, chunk_agg=_MAX).count("%s")
        )

    def test_clip_topk_mean_structure(self) -> None:
        sql = _two_stage_clip_for_ids_sql(1536, chunk_agg=_TOPK)
        self.assertIn("clip_ranked AS (", sql)
        self.assertIn("row_number() OVER (", sql)
        self.assertIn("DESC NULLS LAST, chunk_pos ASC", sql)
        self.assertIn("AVG(CASE WHEN agg_rn <= 3 THEN agg_sim END) AS s_clip", sql)
        self.assertEqual(
            sql.count("%s"), _two_stage_clip_for_ids_sql(1536, chunk_agg=_MAX).count("%s")
        )

    def test_topk_mean_deterministic(self) -> None:
        # 같은 설정 2회 → 동일 SQL 문자열(결정성).
        a = _hybrid_embedding_bm25_sql(1536, chunk_agg=_TOPK)
        b = _hybrid_embedding_bm25_sql(1536, chunk_agg=_TOPK)
        self.assertEqual(a, b)


class TestMixSqlStructure(unittest.TestCase):
    """mix: w*MAX + (1-w)*AVG. 파라미터 개수 불변."""

    def test_hybrid_mix_has_weighted_max_avg(self) -> None:
        sql = _hybrid_embedding_bm25_sql(1536, chunk_agg=ChunkAggConfig("mix", 3, 0.5))
        self.assertIn(
            "(0.5 * MAX(agg_sim) + (1 - 0.5) * AVG(agg_sim)) AS emb_score", sql
        )
        self.assertEqual(
            sql.count("%s"), _hybrid_embedding_bm25_sql(1536, chunk_agg=_MAX).count("%s")
        )

    def test_mix_weight_reflected(self) -> None:
        sql = _two_stage_stage1_sql(1536, chunk_agg=ChunkAggConfig("mix", 3, 0.8))
        self.assertIn("(0.8 * MAX(agg_sim) + (1 - 0.8) * AVG(agg_sim)) AS s_text", sql)


if __name__ == "__main__":
    unittest.main()
