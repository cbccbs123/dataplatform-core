"""006 하이브리드 검색 재배선 — 레거시(media_items/media_chunks) → 현행(asset_*) 스키마.

US1 인수기준: 검색 SQL이 드롭된 레거시 테이블을 참조하지 않고
``asset_metadata``(FTS·요약)·``asset_embedding``(채널·청크 1536D)만 사용한다.
SQL 빌더는 문자열을 반환하므로 DB 없이 구조적으로 검증한다(실 동작은 RUN_DB_E2E e2e).
"""

from __future__ import annotations

import math
import os
import unittest
import uuid
from pathlib import Path
from unittest import mock

from dotenv import load_dotenv

from src.search import media_search as ms

_RUN_DB = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


def _unit_vec(idx: int) -> list[float]:
    """idx 위치만 1.0 인 1536D 단위 벡터(직교 쌍·코사인 통제용)."""
    from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION

    v = [0.0] * FIX_EMBEDDING_DIMENSION
    v[idx] = 1.0
    return v


def _make_asset_with_embeddings(
    db: object,
    ids: list,
    *,
    fts: str,
    summary: str,
    modality: str,
    embeddings: list[tuple[str, list[float], int]],
) -> str:
    """registered 자산 1건 생성. embeddings=[(channel, vector, chunk_index), ...]."""
    from src.dispatch.types import AssetRecord, EmbeddingItem
    from src.ingest.status import AssetStatus, set_status
    from src.registry.asset_persist import create_asset, finalize_asset

    with db.transaction() as conn:
        aid = create_asset(
            conn, fs_path=f"/t/{uuid.uuid4().hex}.bin", modality=modality, file_hash=uuid.uuid4().hex
        )
    ids.append(aid)
    with db.transaction() as conn:
        set_status(conn, aid, AssetStatus.ROUTING)
        set_status(conn, aid, AssetStatus.CLASSIFYING)
        set_status(conn, aid, AssetStatus.EXTRACTING)
    items = [
        EmbeddingItem(channel=ch, vector=vec, model_name="test", chunk_index=ci)
        for ch, vec, ci in embeddings
    ]
    with db.transaction() as conn:
        finalize_asset(
            conn,
            aid,
            AssetRecord(ext_meta={"summary": summary}, fts_plain=fts, embeddings=items),
        )
    return str(aid)


def _make_search_asset(
    db: object,
    ids: list,
    *,
    fts: str,
    summary: str,
    st_vector: list[float],
    modality: str = "txt",
) -> str:
    """registered 텍스트 자산 1건: search_vector(fts) + ext_meta.summary + st 임베딩(chunk 0)."""
    return _make_asset_with_embeddings(
        db, ids, fts=fts, summary=summary, modality=modality, embeddings=[("st", st_vector, 0)]
    )

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
        self.assertIn("fs_path", sql)  # file_uri ← asset.fs_path(실제 채워지는 경로; fs_uri는 항상 NULL)


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

    def test_coerces_uuid_id_to_str(self) -> None:
        # psycopg 는 asset_id 를 uuid.UUID 로 반환 → 결과 계약(시각 경로와 동일)상 str 로 통일.
        u = uuid.UUID("018f0000-0000-7000-8000-000000000020")
        out = ms._sanitize_hybrid_search_rows([{"id": u, "similarity": 0.5}])
        self.assertIsInstance(out[0]["id"], str)
        self.assertEqual(out[0]["id"], str(u))


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


class TestMergeClipOnlyCandidates(unittest.TestCase):
    """ST(1차) 후보 + CLIP-단독 후보 병합 시 동일 자산 id 중복 제거(T019/#7)."""

    def test_existing_id_not_duplicated_new_id_added(self) -> None:
        merged = {
            "x": {"id": "x", "file_uri": "/x", "modality": "video", "s_text": 0.9, "s_clip": 0.1}
        }
        clip_extra = [
            {"id": "x", "file_uri": "/x", "modality": "video", "similarity": 0.8},  # 중복 → 무시
            {"id": "y", "file_uri": "/y", "modality": "image", "similarity": 0.7},  # 신규 → 추가
        ]
        out = ms._merge_clip_only_candidates(merged, clip_extra)
        self.assertEqual(set(out.keys()), {"x", "y"})
        self.assertEqual(out["x"]["s_text"], 0.9)  # 기존 ST 값 보존(덮어쓰지 않음)
        self.assertEqual(out["y"]["s_text"], 0.0)  # CLIP-단독 → s_text 0
        self.assertAlmostEqual(out["y"]["s_clip"], 0.7)


class TestDedupeById(unittest.TestCase):
    """여러 source 병합 시 자산 id 당 1건(최고 similarity)만 — video 버킷 ST+시각 중복 제거."""

    def test_keeps_highest_similarity_per_id(self) -> None:
        rows = [
            {"id": "v1", "similarity": 0.4, "source": "visual_two_stage"},
            {"id": "v1", "similarity": 0.9, "source": "st_hybrid"},
            {"id": "v2", "similarity": 0.5, "source": "st_hybrid"},
        ]
        out = ms._dedupe_by_id(rows)
        by_id = {r["id"]: r for r in out}
        self.assertEqual(set(by_id), {"v1", "v2"})  # 중복 v1 → 1건
        self.assertEqual(by_id["v1"]["similarity"], 0.9)  # 높은 쪽 보존
        self.assertEqual(by_id["v1"]["source"], "st_hybrid")


@unittest.skipUnless(_RUN_DB, "RUN_DB_E2E=1 일 때만(실 PostgreSQL)")
class TestHybridSearchRewireDB(unittest.TestCase):
    """재배선된 하이브리드 SQL이 실 DB(asset_*)에서 동작 — 레거시 참조 없이 결과 반환(T007/T008)."""

    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(_ENV, override=False)
        from src.database.postgres_util import PostgresUtil

        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db.__exit__(None, None, None)

    def setUp(self) -> None:
        self._ids: list = []

    def tearDown(self) -> None:
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def test_text_query_returns_current_schema_result(self) -> None:  # T007
        vec = _unit_vec(0)
        aid = _make_search_asset(
            self.db, self._ids, fts="워크숍 발표 자료", summary="작년 워크숍 발표", st_vector=vec
        )
        rows = ms._run_hybrid_search(
            query_vector=vec,
            bm25_query="워크숍",
            media_types=["txt"],
            embedding_kind="st",
            limit=10,
            alpha=0.75,
        )
        ids = [r["id"] for r in rows]
        self.assertIn(aid, ids)  # 레거시 참조 없이 현행 스키마로 매칭
        top = next(r for r in rows if r["id"] == aid)
        self.assertEqual(top["modality"], "txt")
        self.assertIn("file_uri", top)
        self.assertEqual(top["summary"], "작년 워크숍 발표")
        self.assertGreater(top["similarity"], 0.0)

    def test_irrelevant_query_finite_near_zero_no_exception(self) -> None:  # T008
        _make_search_asset(self.db, self._ids, fts="고양이 사진", summary="", st_vector=_unit_vec(0))
        rows = ms._run_hybrid_search(
            query_vector=_unit_vec(1),  # 저장 벡터와 직교
            bm25_query="전혀무관한질의어휘zzz",
            media_types=["txt"],
            embedding_kind="st",
            limit=10,
            alpha=0.75,
        )
        self.assertIsInstance(rows, list)
        for r in rows:
            self.assertTrue(math.isfinite(r["similarity"]))
            self.assertLess(r["similarity"], 0.2)  # 무관 → 0 근처

    def test_blank_query_no_exception(self) -> None:  # T008
        _make_search_asset(self.db, self._ids, fts="문서", summary="", st_vector=_unit_vec(0))
        rows = ms._run_hybrid_search(
            query_vector=_unit_vec(2),
            bm25_query="",
            media_types=["txt"],
            embedding_kind="st",
            limit=10,
            alpha=0.75,
        )
        self.assertIsInstance(rows, list)  # 빈/공백 질의도 예외 없음

    def test_alpha_boundary_monotonic_ranking(self) -> None:  # T009 단조성
        q = _unit_vec(0)
        # A: FTS 강함(bm25 매치) / 벡터 약함(직교).  B: 벡터 강함(=query) / bm25 없음.
        a = _make_search_asset(
            self.db, self._ids, fts="알파베타감마델타", summary="", st_vector=_unit_vec(7)
        )
        b = _make_search_asset(
            self.db, self._ids, fts="무관한다른텍스트", summary="", st_vector=q
        )
        common = {
            "query_vector": q,
            "bm25_query": "알파베타감마델타",
            "media_types": ["txt"],
            "embedding_kind": "st",
            "limit": 50,
        }
        rank_vec = [r["id"] for r in ms._run_hybrid_search(alpha=1.0, **common)]
        rank_fts = [r["id"] for r in ms._run_hybrid_search(alpha=0.0, **common)]
        # alpha=1(벡터 단독) → 벡터매치 B 가 FTS매치 A 보다 위
        self.assertLess(rank_vec.index(b), rank_vec.index(a))
        # alpha=0(FTS 단독) → FTS매치 A 가 벡터매치 B 보다 위
        self.assertLess(rank_fts.index(a), rank_fts.index(b))

    def test_single_keyframe_video_not_dropped(self) -> None:  # T017/T018 — chunk_index=0 누락 없음
        from psycopg.rows import dict_row

        vid = _make_asset_with_embeddings(
            self.db,
            self._ids,
            fts="해변 영상",
            summary="여름 해변",
            modality="video",
            embeddings=[("st", _unit_vec(0), 0), ("clip", _unit_vec(1), 0)],  # 단일 키프레임 index 0
        )
        # 1차(ST): chunk_index=0 단일 청크 영상이 MAX 집계에서 누락되지 않고 후보로 잡혀야 한다.
        with self.db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                ms._two_stage_stage1_sql(1536),
                (_unit_vec(0), ["image", "video"], "st", 50),
            )
            stage1_ids = [str(r["id"]) for r in cur.fetchall()]
        self.assertIn(vid, stage1_ids)
        # 2차(CLIP): 같은 영상의 clip(chunk_index=0) 청크도 id 목록 조회에서 잡혀야 한다.
        with self.db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                ms._two_stage_clip_for_ids_sql(1536),
                (_unit_vec(1), ["image", "video"], "clip", [vid]),
            )
            clip_ids = [str(r["id"]) for r in cur.fetchall()]
        self.assertIn(vid, clip_ids)

    def test_special_chars_query_no_tsquery_error(self) -> None:  # T020 tsquery 안전화
        _make_search_asset(self.db, self._ids, fts="보고서 자료", summary="", st_vector=_unit_vec(0))
        for q in ["a & b | c", "( ) : !", "검색 & 자료 | (보고서)", "C++ & 100% : test!"]:
            with self.subTest(q=q):
                rows = ms._run_hybrid_search(
                    query_vector=_unit_vec(0),
                    bm25_query=q,
                    media_types=["txt"],
                    embedding_kind="st",
                    limit=10,
                    alpha=0.5,
                )
                self.assertIsInstance(rows, list)  # to_tsquery 구문 오류 없이 처리


class TestGroupedIncludeVisual(unittest.TestCase):
    """``search_media_all_grouped(include_visual=...)`` 가 시각 2단계(CLIP) 실행 여부를 제어한다.

    DB·모델을 때리는 ``search_media_text_items``/``search_media_images_two_stage`` 를 모킹해
    호출 여부만 검증한다(텍스트/오디오만 요청 시 CLIP 미실행 = 낭비 제거, 결과 동치).
    """

    _STRUCTURED = {"semantic_query": "회식", "semantic_query_en": "dinner"}

    def test_include_visual_false_skips_two_stage(self) -> None:
        with mock.patch.object(ms, "search_media_text_items", return_value=[]) as mt, \
             mock.patch.object(ms, "search_media_images_two_stage", return_value=[]) as mv:
            out = ms.search_media_all_grouped(
                "회식", structured=self._STRUCTURED, include_visual=False
            )
        mt.assert_called_once()  # ST 하이브리드(텍스트/오디오/영상텍스트)는 여전히 실행
        mv.assert_not_called()   # 시각 2단계(CLIP)는 건너뜀
        self.assertEqual(out["image"], [])  # 이미지 버킷은 비어야(시각 미실행)

    def test_include_visual_true_calls_two_stage(self) -> None:
        with mock.patch.object(ms, "search_media_text_items", return_value=[]) as mt, \
             mock.patch.object(ms, "search_media_images_two_stage", return_value=[]) as mv:
            ms.search_media_all_grouped(
                "회식", structured=self._STRUCTURED, include_visual=True
            )
        mt.assert_called_once()
        mv.assert_called_once()  # 시각 2단계 실행

    def test_default_keeps_visual_backward_compatible(self) -> None:
        # include_visual 미지정(기본 True) → 기존 동작과 동일하게 시각 2단계 실행(회귀 0).
        with mock.patch.object(ms, "search_media_text_items", return_value=[]), \
             mock.patch.object(ms, "search_media_images_two_stage", return_value=[]) as mv:
            ms.search_media_all_grouped("회식", structured=self._STRUCTURED)
        mv.assert_called_once()

    def test_include_visual_false_video_bucket_keeps_st_only(self) -> None:
        # include_visual=False 라도 ST 하이브리드 영상(video_st)은 보존되고 시각 혼입만 빠진다.
        # → video 버킷이 텅 비는 게 아니라 video_st 만으로 구성됨을 grouped 레벨에서 직접 확인.
        video_st_row = {"id": "v1", "similarity": 0.8, "modality": ms.MediaKind.VIDEO.value}
        with mock.patch.object(ms, "search_media_text_items", return_value=[video_st_row]), \
             mock.patch.object(ms, "search_media_images_two_stage", return_value=[]) as mv:
            out = ms.search_media_all_grouped(
                "회식", structured=self._STRUCTURED, include_visual=False
            )
        mv.assert_not_called()
        self.assertEqual([r["id"] for r in out["video"]], ["v1"])  # video_st 보존
        self.assertEqual([r["source"] for r in out["video"]], ["st_hybrid"])
        self.assertEqual(out["image"], [])  # 시각 미실행 → 이미지 빈 버킷


if __name__ == "__main__":
    unittest.main()
