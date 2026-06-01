"""006 하이브리드 검색 재배선 — 레거시(media_items/media_chunks) → 현행(asset_*) 스키마.

US1 인수기준: 검색 SQL이 드롭된 레거시 테이블을 참조하지 않고
``asset_metadata``(FTS·요약)·``asset_embedding``(채널·청크 1536D)만 사용한다.
SQL 빌더는 문자열을 반환하므로 DB 없이 구조적으로 검증한다(실 동작은 RUN_DB_E2E e2e).
"""

from __future__ import annotations

import unittest

from src.search import media_search as ms

# 드롭된 v1 레거시 식별자 — 어떤 검색 SQL에도 남아 있으면 안 된다.
_LEGACY_TOKENS = (
    "media_items",
    "media_chunks",
    "media_item_id",
    "embedding_kind",
    "media_type",
    "metadata->>'summary'",
)


def _all_search_sql() -> dict[str, str]:
    """검색에 쓰이는 모든 SQL 표면(빌더 결과 + 모듈 상수)."""
    return {
        "hybrid": ms._hybrid_embedding_bm25_sql(1536),
        "two_stage_stage1": ms._two_stage_stage1_sql(1536),
        "two_stage_clip": ms._two_stage_clip_for_ids_sql(1536),
        "two_stage_bm25_for_ids": ms._TWO_STAGE_BM25_FOR_IDS_SQL,
        "summaries": ms._SUMMARIES_FOR_ASSET_IDS_SQL,
    }


class TestNoLegacyTableReferences(unittest.TestCase):
    def test_no_legacy_tokens_in_any_search_sql(self) -> None:
        for name, sql in _all_search_sql().items():
            for tok in _LEGACY_TOKENS:
                self.assertNotIn(
                    tok, sql, f"{name} SQL에 레거시 식별자 '{tok}' 가 남아 있음"
                )


class TestHybridSqlCurrentSchema(unittest.TestCase):
    def test_hybrid_sql_references_asset_schema(self) -> None:
        sql = ms._hybrid_embedding_bm25_sql(1536)
        self.assertIn("asset_embedding", sql)
        self.assertIn("asset_metadata", sql)
        self.assertIn("channel", sql)  # embedding_kind → channel
        self.assertIn("ext_meta", sql)  # summary 출처
        self.assertIn("modality", sql)  # media_type → modality
        self.assertIn("fs_uri", sql)  # file_uri → asset.fs_uri


class TestFiniteFloat(unittest.TestCase):
    """드라이버/JSON 에서 오는 NaN·inf·NULL 을 유한 실수로 안전화(T010)."""

    def test_none_returns_default(self) -> None:
        self.assertEqual(ms._finite_float(None, 0.0), 0.0)
        self.assertEqual(ms._finite_float(None, -1.0), -1.0)

    def test_nan_and_inf_return_default(self) -> None:
        self.assertEqual(ms._finite_float(float("nan"), 0.0), 0.0)
        self.assertEqual(ms._finite_float(float("inf"), 0.0), 0.0)
        self.assertEqual(ms._finite_float(float("-inf"), 0.0), 0.0)

    def test_non_numeric_returns_default(self) -> None:
        self.assertEqual(ms._finite_float("not-a-number", 0.0), 0.0)

    def test_finite_value_passthrough(self) -> None:
        self.assertEqual(ms._finite_float(0.42, 0.0), 0.42)
        self.assertEqual(ms._finite_float("0.5", 0.0), 0.5)


class TestSaturatingBm25(unittest.TestCase):
    """BM25 포화 정규화 bm25/(bm25+k): 단조 증가·[0,1) 범위·k에서 0.5(T010)."""

    def test_zero_is_zero(self) -> None:
        self.assertEqual(ms._saturating_bm25(0.0), 0.0)

    def test_at_k_is_half(self) -> None:
        self.assertAlmostEqual(ms._saturating_bm25(ms.BM25_SATURATION_K), 0.5)

    def test_monotonic_increasing_and_bounded(self) -> None:
        vals = [ms._saturating_bm25(x) for x in (0.0, 0.1, 0.5, 1.0, 10.0, 1000.0)]
        self.assertEqual(vals, sorted(vals))  # 단조 증가
        for v in vals:
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)  # 포화 상한 < 1

    def test_negative_clamped_to_zero(self) -> None:
        self.assertEqual(ms._saturating_bm25(-5.0), 0.0)


class TestSanitizeHybridRows(unittest.TestCase):
    """결과 행의 점수 필드 NaN/inf/NULL 보정 + candidate_count int 화(T010)."""

    def test_sanitizes_score_fields(self) -> None:
        rows = [
            {
                "similarity": float("nan"),
                "emb_score": float("inf"),
                "bm25_score": None,
                "bm25_scaled": "bad",
                "candidate_count": "3",
            }
        ]
        out = ms._sanitize_hybrid_search_rows(rows)
        self.assertEqual(out[0]["similarity"], 0.0)
        self.assertEqual(out[0]["emb_score"], 0.0)
        self.assertEqual(out[0]["bm25_score"], 0.0)
        self.assertEqual(out[0]["bm25_scaled"], 0.0)
        self.assertEqual(out[0]["candidate_count"], 3)


class TestDeterministicTiebreak(unittest.TestCase):
    """동점 similarity 에서 입력 순서에 의존하지 않는 결정적 순서(T011 — 헌법 3조)."""

    def test_sort_by_similarity_cap_tiebreak_by_id(self) -> None:
        a = {"id": "bbb", "similarity": 0.5}
        b = {"id": "aaa", "similarity": 0.5}
        out1 = [r["id"] for r in ms._sort_by_similarity_cap([a, b], 10)]
        out2 = [r["id"] for r in ms._sort_by_similarity_cap([b, a], 10)]
        self.assertEqual(out1, out2)  # 입력 순서 무관 동일 출력
        self.assertEqual(out1, ["aaa", "bbb"])  # id 오름차순 tiebreak

    def test_hybrid_sql_order_by_has_id_tiebreak(self) -> None:
        sql = ms._hybrid_embedding_bm25_sql(1536)
        self.assertIn("ORDER BY similarity DESC NULLS LAST, id", sql)


class TestHybridSearchAlphaRange(unittest.TestCase):
    """alpha 범위 검증은 DB 접근 이전에 수행되어야 한다(T009 — 범위부분)."""

    def test_alpha_above_one_raises_before_db(self) -> None:
        with self.assertRaises(ValueError):
            ms._run_hybrid_search(
                query_vector=[0.0],
                bm25_query="q",
                media_types=["txt"],
                embedding_kind="st",
                limit=10,
                alpha=1.5,
            )

    def test_alpha_below_zero_raises_before_db(self) -> None:
        with self.assertRaises(ValueError):
            ms._run_hybrid_search(
                query_vector=[0.0],
                bm25_query="q",
                media_types=["txt"],
                embedding_kind="st",
                limit=10,
                alpha=-0.1,
            )

    def test_two_stage_alpha_range_raises_before_db(self) -> None:
        with self.assertRaises(ValueError):
            ms.search_media_images_two_stage("질의", "query", alpha=2.0)
        with self.assertRaises(ValueError):
            ms.search_media_images_two_stage("질의", "query", bm25_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
