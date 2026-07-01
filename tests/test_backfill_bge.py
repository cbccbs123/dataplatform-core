"""BGE-M3 백필 스크립트 단위/e2e 테스트 (017 G2 — ext_meta 기반 재설계).

017 A/B 는 같은 자산을 KoSimCSE(``channel='st'``)와 BGE-M3(``channel='st_bge'``) 두 채널로
임베딩한다. 백필 스크립트는 이미 ``channel='st'`` 행이 있는 registered 자산의 **'st' 본문을
modality 별로 재현**해(VLM/CLIP/whisper 재실행 없이 ``ext_meta``·``fs_path`` 산출물 재사용)
BGE 임베딩을 ``channel='st_bge'`` 행으로 추가한다(스키마 무변경).

본문 재현 분기(ingest ``_embed_*`` 와 동일 입력 — chunk_index 정합):
  - **문서**(txt/json/pdf/word/excel/powerpoint): ``embedding_text_chunks(fs_path, file_kind=…)``.
  - **image**: ``build_image_vlm_text_for_embedding(core_meta|ext_meta)`` → 1청크(chunk_index=0).
  - **video**: ``ext_meta.keyframes`` 프레임별 ``build_image_vlm_text_for_embedding`` → chunk_index=프레임순번.
  - **audio**: ``ext_meta.stt``(없으면 ``.stt.txt`` 사이드카) → ``embedding_plain_text_chunks``.

테스트 전략(docs/테스트_가이드.md §2~3)
  - **mock conn 순수 단위(RED-first)**: SQL 배선(core/ext 조회·INSERT)·modality 본문 분기·
    본문 누락 skip(예외 전파 안 함)·배치 격리를 실 DB/모델 없이 검증한다. 무거운 임베딩
    (``embed_texts``/``embedding_*_chunks``)은 mock 으로 대체하고, 순수 텍스트 빌더는 실제 실행.
  - **실 DB e2e(``RUN_DB_E2E`` 게이트)**: 픽스처 자산(문서·image·audio) → 백필 → ``st``·``st_bge``
    공존(chunk 수 일치) + 2회 멱등(중복 0). BGE-M3 모델 다운로드·GPU·dev DB 필요(사람 실행).

백필 스크립트는 ``scripts/`` 에 있고 패키지가 아니므로 importlib 로 파일에서 직접 적재한다.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import ModuleType
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]
_MOD_PATH = _REPO / "scripts" / "backfill_bge_embeddings.py"
_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = _REPO / ".env.dev"
_BGE = "BAAI/bge-m3"


def _load_backfill_module() -> ModuleType:
    """``scripts/backfill_bge_embeddings.py`` 를 모듈로 적재(scripts 는 패키지 아님)."""
    spec = importlib.util.spec_from_file_location("backfill_bge_embeddings", _MOD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mock_conn() -> tuple[mock.MagicMock, mock.MagicMock]:
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    return conn, cur


class _FakeDB:
    """``run_backfill_db`` 용 최소 PostgresUtil 대역 — ``transaction()`` 컨텍스트만 제공."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    @contextlib.contextmanager
    def transaction(self):  # noqa: ANN201 — 테스트 대역
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = self.rows
        yield conn


class TestFetchStAssets(unittest.TestCase):
    """대상 조회: channel='st' + registered 자산의 asset_id·fs_path·modality·core/ext_meta."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def test_query_filters_registered_st_assets_with_meta(self) -> None:
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [
            {"asset_id": "a1", "fs_path": "/d/x.txt", "modality": "txt",
             "core_meta": {}, "ext_meta": {}}
        ]
        rows = self.backfill._fetch_st_assets(conn, limit=5)
        sql = cur.execute.call_args.args[0]
        self.assertIn("asset_embedding", sql)
        self.assertIn("'st'", sql)          # channel='st' 가 있는 자산만
        self.assertIn("registered", sql)    # status='registered' 만
        self.assertIn("asset_metadata", sql)  # core/ext_meta 를 함께 조회(modality 본문 재현용)
        self.assertIn("core_meta", sql)
        self.assertIn("ext_meta", sql)
        self.assertIn("LIMIT", sql)         # --limit 단계적 백필
        self.assertIn(5, cur.execute.call_args.args[1])
        self.assertEqual(rows[0]["modality"], "txt")

    def test_no_limit_omits_limit_clause(self) -> None:
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = []
        self.backfill._fetch_st_assets(conn, limit=None)
        sql = cur.execute.call_args.args[0]
        self.assertNotIn("LIMIT", sql)


class TestBackfillAssetDocument(unittest.TestCase):
    """문서 자산: fs_path 파일을 ``embedding_text_chunks`` 로 청킹+임베딩(현행 유지)."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def _call(self, conn, row):
        return self.backfill.backfill_asset(
            conn, row, model_name=_BGE, chunk_size=512, encoding="utf-8", normalize=True
        )

    def test_document_modality_uses_embedding_text_chunks_and_inserts_st_bge(self) -> None:
        conn, cur = _mock_conn()
        chunks = [
            {"chunk_index": 0, "embedding_vector": [0.1] * 1536},
            {"chunk_index": 1, "embedding_vector": [0.2] * 1536},
        ]
        # 053: 저장 modality 가 canonical 'text'. _document_file_kind('text')=None 이므로
        # detect_file_kind(fs_path) 로 세분류(txt)를 재도출해 문서 분기를 타야 한다(FR-505).
        # detect_file_kind 는 실 파일 libmagic 판정이므로 seam 으로 mock(파일 부재여도 검증 가능).
        with mock.patch.object(self.backfill, "embedding_text_chunks", return_value=chunks) as m_doc, \
             mock.patch.object(self.backfill, "embedding_plain_text_chunks") as m_plain, \
             mock.patch.object(self.backfill, "detect_file_kind", return_value="txt"):
            n = self._call(
                conn,
                {"asset_id": "a1", "fs_path": "/d/x.txt", "modality": "text",
                 "core_meta": {}, "ext_meta": {}},
            )

        self.assertEqual(n, 2)
        m_doc.assert_called_once()
        m_plain.assert_not_called()  # 문서 분기는 plain 미사용
        # 본문은 fs_path 에서 재로딩, 재도출한 세분류 file_kind='txt', BGE 모델로 임베딩.
        self.assertEqual(m_doc.call_args.kwargs["file_kind"], "txt")
        self.assertEqual(m_doc.call_args.kwargs["embedding_model_name"], _BGE)

        self.assertEqual(cur.executemany.call_count, 1)
        sql = cur.executemany.call_args.args[0]
        self.assertIn("INSERT INTO asset_embedding", sql)
        # 멱등: 같은 (asset_id, channel, chunk_index) 재실행 시 중복 0(SC-001).
        self.assertIn("ON CONFLICT (asset_id, channel, chunk_index) DO NOTHING", sql)
        rows = cur.executemany.call_args.args[1]
        # (asset_id, channel, chunk_index, embedding, model_name, model_version) — channel 은 파라미터.
        self.assertEqual([r[1] for r in rows], ["st_bge", "st_bge"])
        self.assertEqual([r[2] for r in rows], [0, 1])
        self.assertEqual([r[4] for r in rows], [_BGE, _BGE])

    def test_legacy_file_kind_modality_still_uses_document_branch(self) -> None:
        # 053 하위호환: 마이그레이션 전 구 저장값('txt'/'json' 등)도 계속 문서 분기를 탄다.
        # _document_file_kind('json')='json' 이라 detect_file_kind 재도출 없이 그대로 동작.
        conn, cur = _mock_conn()
        chunks = [{"chunk_index": 0, "embedding_vector": [0.1] * 1536}]
        with mock.patch.object(self.backfill, "embedding_text_chunks", return_value=chunks) as m_doc, \
             mock.patch.object(self.backfill, "detect_file_kind") as m_detect:
            n = self._call(
                conn,
                {"asset_id": "a2", "fs_path": "/d/x.json", "modality": "json",
                 "core_meta": {}, "ext_meta": {}},
            )
        self.assertEqual(n, 1)
        m_detect.assert_not_called()  # 구값은 _document_file_kind 로 이미 해소 → 재도출 불필요
        self.assertEqual(m_doc.call_args.kwargs["file_kind"], "json")

    def test_missing_document_body_skips_without_raising(self) -> None:
        conn, cur = _mock_conn()
        # 파일 부재 → embedding_text_chunks 가 FileNotFoundError. 백필은 예외를 흡수하고 skip.
        with mock.patch.object(
            self.backfill, "embedding_text_chunks", side_effect=FileNotFoundError("/d/gone.txt")
        ):
            n = self._call(
                conn,
                {"asset_id": "a3", "fs_path": "/d/gone.txt", "modality": "txt",
                 "core_meta": {}, "ext_meta": {}},
            )
        self.assertEqual(n, 0)  # 0 = skip
        cur.executemany.assert_not_called()  # INSERT 안 함


class TestBackfillAssetImage(unittest.TestCase):
    """image 자산: ext_meta(summary+keywords+labels) → build_image_vlm_text 1청크(_embed_image 동일)."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def _call(self, conn, row):
        return self.backfill.backfill_asset(
            conn, row, model_name=_BGE, chunk_size=512, encoding="utf-8", normalize=True
        )

    def test_image_uses_vlm_text_builder_single_chunk(self) -> None:
        conn, cur = _mock_conn()
        core_meta = {"width": 100, "height": 80}
        ext_meta = {
            "summary": "무선 충전기 제품 이미지",
            "keywords": ["무선 충전기", "Qi2", "UGREEN"],
            "labels": [{"label": "충전기", "score": 0.5}, {"label": "손", "score": 0.2}],
        }
        # 순수 빌더는 실제 실행 — 같은 입력(core|ext)으로 _embed_image 와 동일한 텍스트가 나온다.
        expected_text = self.backfill.build_image_vlm_text_for_embedding({**core_meta, **ext_meta})
        with mock.patch.object(self.backfill, "embed_texts", return_value=[[0.1] * 1024]) as m_embed, \
             mock.patch.object(self.backfill, "embedding_text_chunks") as m_doc, \
             mock.patch.object(self.backfill, "embedding_plain_text_chunks") as m_plain:
            n = self._call(
                conn,
                {"asset_id": "img1", "fs_path": "/d/p.jpg", "modality": "image",
                 "core_meta": core_meta, "ext_meta": ext_meta},
            )

        self.assertEqual(n, 1)  # 이미지는 단일 청크
        m_doc.assert_not_called()
        m_plain.assert_not_called()
        # build_image_vlm_text_for_embedding(core|ext) 결과를 BGE 로 임베딩(_embed_image 와 동일).
        m_embed.assert_called_once()
        self.assertEqual(m_embed.call_args.args[0], [expected_text])
        self.assertEqual(m_embed.call_args.kwargs["model_name"], _BGE)
        rows = cur.executemany.call_args.args[1]
        self.assertEqual([r[1] for r in rows], ["st_bge"])
        self.assertEqual([r[2] for r in rows], [0])
        # pad 로 1536D 저장 차원 보존(1024→1536).
        self.assertEqual(len(rows[0][3]), 1536)

    def test_image_with_empty_ext_meta_skips(self) -> None:
        # summary·keywords·labels 가 없으면 본문이 공백뿐 → skip(garbage 임베딩 방지).
        conn, cur = _mock_conn()
        with mock.patch.object(self.backfill, "embed_texts") as m_embed:
            n = self._call(
                conn,
                {"asset_id": "img2", "fs_path": "/d/blank.jpg", "modality": "image",
                 "core_meta": {"width": 10}, "ext_meta": {}},
            )
        self.assertEqual(n, 0)
        m_embed.assert_not_called()
        cur.executemany.assert_not_called()


class TestBackfillAssetVideo(unittest.TestCase):
    """video 자산: ext_meta.keyframes 프레임별 build_image_vlm_text 다청크(_embed_video 동일)."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def _call(self, conn, row):
        return self.backfill.backfill_asset(
            conn, row, model_name=_BGE, chunk_size=512, encoding="utf-8", normalize=True
        )

    def test_video_keyframes_produce_one_chunk_per_frame(self) -> None:
        conn, cur = _mock_conn()
        # DB ext_meta.keyframes 실제 구조: 각 항목 summary 가 dict({summary,keywords,objects}),
        # labels 가 list[{label,score}]. _embed_video 는 이를 프레임 메타로 평탄화한다.
        ext_meta = {
            "keyframes": [
                {"scene_index": 1, "start_sec": 0.0, "end_sec": 5.0,
                 "summary": {"summary": "프레임0 설명", "keywords": ["a", "b"], "objects": ["x"]},
                 "labels": [{"label": "충전기", "score": 0.3}]},
                {"scene_index": 2, "start_sec": 5.0, "end_sec": 9.0,
                 "summary": {"summary": "프레임1 설명", "keywords": [], "objects": []},
                 "labels": []},
            ]
        }
        # _embed_video 와 동일한 프레임 메타 평탄화로 기대 텍스트 산정.
        f0 = self.backfill.build_image_vlm_text_for_embedding(
            {"summary": "프레임0 설명", "keywords": ["a", "b"], "labels": [{"label": "충전기", "score": 0.3}]}
        )
        with mock.patch.object(self.backfill, "embed_texts", return_value=[[0.2] * 1024]) as m_embed:
            n = self._call(
                conn,
                {"asset_id": "v1", "fs_path": "/d/v.mp4", "modality": "video",
                 "core_meta": {}, "ext_meta": ext_meta},
            )

        self.assertEqual(n, 2)  # 키프레임 2개 → 2청크
        self.assertEqual(m_embed.call_count, 2)  # 프레임당 1회 임베딩
        self.assertEqual(m_embed.call_args_list[0].args[0], [f0])
        rows = cur.executemany.call_args.args[1]
        self.assertEqual([r[1] for r in rows], ["st_bge", "st_bge"])
        self.assertEqual([r[2] for r in rows], [0, 1])  # chunk_index = 프레임 순번(0-based)

    def test_video_without_keyframes_falls_back_to_asset_summary(self) -> None:
        # keyframes 없으면 자산 summary 1청크로 폴백.
        conn, cur = _mock_conn()
        ext_meta = {"summary": "영상 전체 요약", "keywords": ["요가"]}
        expected = self.backfill.build_image_vlm_text_for_embedding(ext_meta)
        with mock.patch.object(self.backfill, "embed_texts", return_value=[[0.3] * 1024]) as m_embed:
            n = self._call(
                conn,
                {"asset_id": "v2", "fs_path": "/d/v2.mp4", "modality": "video",
                 "core_meta": {}, "ext_meta": ext_meta},
            )
        self.assertEqual(n, 1)
        self.assertEqual(m_embed.call_args.args[0], [expected])
        self.assertEqual([r[2] for r in cur.executemany.call_args.args[1]], [0])

    def test_video_without_any_body_skips(self) -> None:
        # keyframes 도 summary 도 없으면 skip.
        conn, cur = _mock_conn()
        with mock.patch.object(self.backfill, "embed_texts") as m_embed:
            n = self._call(
                conn,
                {"asset_id": "v3", "fs_path": "/d/v3.mp4", "modality": "video",
                 "core_meta": {}, "ext_meta": {}},
            )
        self.assertEqual(n, 0)
        m_embed.assert_not_called()
        cur.executemany.assert_not_called()


class TestBackfillAssetAudio(unittest.TestCase):
    """audio 자산: ext_meta.stt(없으면 .stt.txt 사이드카) → embedding_plain_text_chunks."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def _call(self, conn, row):
        return self.backfill.backfill_asset(
            conn, row, model_name=_BGE, chunk_size=512, encoding="utf-8", normalize=True
        )

    def test_audio_uses_ext_meta_stt(self) -> None:
        conn, cur = _mock_conn()
        chunks = [{"chunk_index": 0, "embedding_vector": [0.3] * 1536}]
        with mock.patch.object(self.backfill, "embedding_plain_text_chunks", return_value=chunks) as m_plain, \
             mock.patch.object(self.backfill, "embedding_text_chunks") as m_doc:
            n = self._call(
                conn,
                {"asset_id": "au1", "fs_path": "/d/a.mp3", "modality": "audio",
                 "core_meta": {}, "ext_meta": {"stt": "안녕하세요 테스트 전사 텍스트"}},
            )
        self.assertEqual(n, 1)
        m_doc.assert_not_called()
        m_plain.assert_called_once()
        # ext_meta.stt 문자열을 BGE 로 청킹 임베딩(whisper 재실행 없음).
        self.assertEqual(m_plain.call_args.args[0], "안녕하세요 테스트 전사 텍스트")
        self.assertEqual(m_plain.call_args.kwargs["embedding_model_name"], _BGE)
        self.assertEqual(cur.executemany.call_args.args[1][0][1], "st_bge")

    def test_audio_falls_back_to_stt_sidecar_when_no_ext_stt(self) -> None:
        conn, cur = _mock_conn()
        chunks = [{"chunk_index": 0, "embedding_vector": [0.4] * 1536}]
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "clip.mp3"
            audio.write_bytes(b"\x00")
            (Path(d) / "clip.mp3.stt.txt").write_text("사이드카 전사", encoding="utf-8")
            with mock.patch.object(self.backfill, "embedding_plain_text_chunks", return_value=chunks) as m_plain:
                n = self._call(
                    conn,
                    {"asset_id": "au2", "fs_path": str(audio), "modality": "audio",
                     "core_meta": {}, "ext_meta": {}},
                )
        self.assertEqual(n, 1)
        m_plain.assert_called_once()
        self.assertEqual(m_plain.call_args.args[0], "사이드카 전사")

    def test_audio_without_stt_or_sidecar_skips(self) -> None:
        conn, cur = _mock_conn()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "no_stt.wav"
            audio.write_bytes(b"\x00")
            with mock.patch.object(self.backfill, "embedding_plain_text_chunks") as m_plain:
                n = self._call(
                    conn,
                    {"asset_id": "au3", "fs_path": str(audio), "modality": "audio",
                     "core_meta": {}, "ext_meta": {}},
                )
        self.assertEqual(n, 0)
        m_plain.assert_not_called()
        cur.executemany.assert_not_called()


class TestBackfillAssetUnknown(unittest.TestCase):
    """unknown 등 본문 재현 불가 modality 는 skip(INSERT 안 함)."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def test_unknown_modality_skips(self) -> None:
        conn, cur = _mock_conn()
        n = self.backfill.backfill_asset(
            conn,
            {"asset_id": "u", "fs_path": "/d/x.bin", "modality": "unknown",
             "core_meta": {}, "ext_meta": {}},
            model_name=_BGE, chunk_size=512, encoding="utf-8", normalize=True,
        )
        self.assertEqual(n, 0)
        cur.executemany.assert_not_called()


class TestRunBackfillDbIsolation(unittest.TestCase):
    """배치 격리: 한 자산의 예외가 배치를 멈추지 않고 skip 으로 집계된다(FR-002)."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def test_per_asset_exception_is_isolated(self) -> None:
        db = _FakeDB(
            [
                {"asset_id": "a", "fs_path": "/x.txt", "modality": "txt", "core_meta": {}, "ext_meta": {}},
                {"asset_id": "b", "fs_path": "/y.txt", "modality": "txt", "core_meta": {}, "ext_meta": {}},
            ]
        )
        with mock.patch.object(self.backfill, "backfill_asset", side_effect=[RuntimeError("boom"), 3]):
            res = self.backfill.run_backfill_db(
                db, model_name=_BGE, chunk_size=512, encoding="utf-8", normalize=True
            )
        self.assertEqual(res["processed"], 1)
        self.assertEqual(res["skipped"], 1)
        self.assertEqual(res["chunks"], 3)

    def test_zero_chunk_return_counts_as_skip(self) -> None:
        db = _FakeDB([{"asset_id": "a", "fs_path": "/x.bin", "modality": "unknown", "core_meta": {}, "ext_meta": {}}])
        with mock.patch.object(self.backfill, "backfill_asset", return_value=0):
            res = self.backfill.run_backfill_db(
                db, model_name=_BGE, chunk_size=512, encoding="utf-8", normalize=True
            )
        self.assertEqual(res["processed"], 0)
        self.assertEqual(res["skipped"], 1)


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만 실행하는 실 DB e2e (BGE-M3 모델·GPU 필요)")
class TestBackfillBgeE2E(unittest.TestCase):
    """픽스처 자산(문서·image·audio) → 백필 → st·st_bge 공존(chunk 수 일치) + 2회 멱등(중복 0)."""

    db = None  # type: ignore[assignment]

    @classmethod
    def setUpClass(cls) -> None:
        from dotenv import load_dotenv

        load_dotenv(_ENV, override=False)
        from src.config.settings import init_settings

        init_settings("dev")
        from src.database.postgres_util import PostgresUtil

        try:
            cls.db = PostgresUtil()
            cls.db.__enter__()
            with cls.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.asset')")
                if cur.fetchone()[0] is None:
                    raise unittest.SkipTest("asset 스키마 미적용")
        except unittest.SkipTest:
            raise
        except Exception as exc:  # noqa: BLE001 — 접속 불가 시 skip
            raise unittest.SkipTest(f"DB 미접속: {type(exc).__name__}: {exc}") from None
        cls.backfill = _load_backfill_module()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.db is not None:
            cls.db.__exit__(None, None, None)

    def setUp(self) -> None:
        self._ids: list[uuid.UUID] = []
        self._dir = Path(tempfile.mkdtemp(prefix="backfill_bge_e2e_"))

    def tearDown(self) -> None:
        import shutil

        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset_metadata WHERE asset_id = ANY(%s)", (self._ids,))
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))
        shutil.rmtree(self._dir, ignore_errors=True)

    def _count(self, asset_id, channel: str) -> int:
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM asset_embedding WHERE asset_id=%s AND channel=%s",
                (asset_id, channel),
            )
            return cur.fetchone()[0]

    def _insert_asset(self, modality, fs_path, *, core_meta, ext_meta, n_chunks):
        """더미 'st' 행 + asset_metadata 를 적재한 registered 자산을 만든다(모델 없이)."""
        from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
        from src.config.settings import get_current_settings
        from src.database.ids import uuid7

        cfg = get_current_settings()
        aid = uuid7()
        self._ids.append(aid)
        dummy = [0.0] * FIX_EMBEDDING_DIMENSION
        dummy[0] = 0.5
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO asset (asset_id, modality, fs_path, status) VALUES (%s,%s,%s,'registered')",
                (aid, modality, str(fs_path)),
            )
            cur.execute(
                "INSERT INTO asset_metadata (asset_id, core_meta, ext_meta) VALUES (%s,%s::jsonb,%s::jsonb)",
                (aid, json.dumps(core_meta, ensure_ascii=False), json.dumps(ext_meta, ensure_ascii=False)),
            )
            cur.executemany(
                f"INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name) "
                f"VALUES (%s,'st',%s,%s::vector({FIX_EMBEDDING_DIMENSION}),%s)",
                [(aid, i, dummy, cfg.text_embedding_model) for i in range(n_chunks)],
            )
        return aid

    def _run_backfill(self, aid, modality, fs_path, core_meta, ext_meta) -> int:
        from src.config.settings import get_current_settings

        cfg = get_current_settings()
        row = {"asset_id": aid, "fs_path": str(fs_path), "modality": modality,
               "core_meta": core_meta, "ext_meta": ext_meta}
        with self.db.transaction() as conn:
            return self.backfill.backfill_asset(
                conn, row,
                model_name=cfg.text_embedding_model_bge,
                chunk_size=cfg.text_embedding_chunk_size,
                encoding=cfg.encoding,
                normalize=cfg.text_embedding_normalize,
            )

    def _assert_coexist_idempotent(self, aid, modality, fs_path, core_meta, ext_meta, n_chunks):
        self.assertEqual(self._count(aid, "st"), n_chunks)
        self.assertEqual(self._count(aid, "st_bge"), 0)
        n1 = self._run_backfill(aid, modality, fs_path, core_meta, ext_meta)
        self.assertEqual(n1, n_chunks)
        self.assertEqual(self._count(aid, "st"), n_chunks)       # st 무변경
        self.assertEqual(self._count(aid, "st_bge"), n_chunks)   # 공존, chunk 수 일치
        # 2회차 → ON CONFLICT DO NOTHING 멱등(중복 0, SC-001).
        self._run_backfill(aid, modality, fs_path, core_meta, ext_meta)
        self.assertEqual(self._count(aid, "st_bge"), n_chunks)

    def test_document_backfill_coexist_idempotent(self) -> None:
        from src.config.settings import get_current_settings
        from src.embedders.text_embedder import _iter_nonempty_chunks

        cfg = get_current_settings()
        txt = self._dir / "doc.txt"
        txt.write_text("가나다라마바사 " * 200, encoding="utf-8")
        n_chunks = len(
            list(
                _iter_nonempty_chunks(
                    txt, file_kind="txt", encoding=cfg.encoding, chunk_size=cfg.text_embedding_chunk_size
                )
            )
        ) or 1
        # 053: 저장은 canonical 'text' — backfill 이 fs_path(doc.txt)에서 file_kind='txt' 재도출(G6).
        aid = self._insert_asset("text", txt, core_meta={}, ext_meta={}, n_chunks=n_chunks)
        self._assert_coexist_idempotent(aid, "text", txt, {}, {}, n_chunks)

    def test_image_backfill_coexist_idempotent(self) -> None:
        # image 의 'st' 는 ingest 가 ext_meta(summary+keywords+labels)로 만든 1청크.
        ext_meta = {
            "summary": "무선 충전기 제품 이미지",
            "keywords": ["무선 충전기", "Qi2"],
            "labels": [{"label": "충전기", "score": 0.5}],
        }
        aid = self._insert_asset(
            "image", "/d/p.jpg", core_meta={"width": 100}, ext_meta=ext_meta, n_chunks=1
        )
        self._assert_coexist_idempotent(aid, "image", "/d/p.jpg", {"width": 100}, ext_meta, 1)

    def test_audio_backfill_coexist_idempotent(self) -> None:
        from src.config.settings import get_current_settings
        from src.file.data_loader import iter_plain_text_chunks

        cfg = get_current_settings()
        stt = "안녕하세요 테스트 전사 텍스트입니다. " * 120
        ext_meta = {"stt": stt}
        n_chunks = len(
            [c for c in iter_plain_text_chunks(stt, chunk_size=cfg.text_embedding_chunk_size, overlap_size=0) if c]
        ) or 1
        aid = self._insert_asset("audio", "/d/a.mp3", core_meta={}, ext_meta=ext_meta, n_chunks=n_chunks)
        self._assert_coexist_idempotent(aid, "audio", "/d/a.mp3", {}, ext_meta, n_chunks)


if __name__ == "__main__":
    unittest.main()
