"""BGE-M3 백필 스크립트 단위/e2e 테스트 (017 G2).

017 A/B 는 같은 자산을 KoSimCSE(``channel='st'``)와 BGE-M3(``channel='st_bge'``) 두 채널로
임베딩한다. 백필 스크립트는 이미 ``channel='st'`` 행이 있는 registered 자산의 본문을
``fs_path``/``modality`` 로 재로딩해 BGE 임베딩을 ``channel='st_bge'`` 행으로 추가한다(스키마 무변경).

테스트 전략(docs/테스트_가이드.md §2~3)
  - **mock conn 순수 단위(RED-first)**: SQL 배선(대상 조회·INSERT)·본문 분기(문서 vs STT)·
    본문 누락 skip(예외 전파 안 함)·배치 격리를 실 DB/모델 없이 검증한다.
    무거운 임베딩(``embedding_text_chunks``/``embedding_plain_text_chunks``)은 mock 으로 대체.
  - **실 DB e2e(``RUN_DB_E2E`` 게이트)**: 픽스처 ``st`` 자산 → 백필 → ``st``·``st_bge`` 공존
    (chunk 수 일치) + 2회 멱등(중복 0). BGE-M3 모델 다운로드·GPU·dev DB 필요(사람 실행).

백필 스크립트는 ``scripts/`` 에 있고 패키지가 아니므로 importlib 로 파일에서 직접 적재한다.
"""

from __future__ import annotations

import contextlib
import importlib.util
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
    """대상 조회: channel='st' + registered 자산의 asset_id·fs_path·modality 만."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def test_query_filters_registered_st_assets(self) -> None:
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [
            {"asset_id": "a1", "fs_path": "/d/x.txt", "modality": "txt"}
        ]
        rows = self.backfill._fetch_st_assets(conn, limit=5)
        sql = cur.execute.call_args.args[0]
        self.assertIn("asset_embedding", sql)
        self.assertIn("'st'", sql)          # channel='st' 가 있는 자산만
        self.assertIn("registered", sql)    # status='registered' 만
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


class TestBackfillAsset(unittest.TestCase):
    """단일 자산 백필 핵심 함수: 본문 분기·INSERT 배선·skip(예외 전파 안 함)."""

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
        with mock.patch.object(self.backfill, "embedding_text_chunks", return_value=chunks) as m_doc, \
             mock.patch.object(self.backfill, "embedding_plain_text_chunks") as m_plain:
            n = self._call(conn, {"asset_id": "a1", "fs_path": "/d/x.txt", "modality": "txt"})

        self.assertEqual(n, 2)
        m_doc.assert_called_once()
        m_plain.assert_not_called()  # 문서 분기는 plain 미사용
        # 본문은 fs_path 에서 재로딩, file_kind=modality, BGE 모델로 임베딩.
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

    def test_audio_modality_uses_plain_chunks_from_stt_sidecar(self) -> None:
        conn, cur = _mock_conn()
        chunks = [{"chunk_index": 0, "embedding_vector": [0.3] * 1536}]
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "clip.mp3"
            audio.write_bytes(b"\x00")
            (Path(d) / "clip.mp3.stt.txt").write_text("안녕하세요 테스트 전사", encoding="utf-8")
            with mock.patch.object(self.backfill, "embedding_plain_text_chunks", return_value=chunks) as m_plain, \
                 mock.patch.object(self.backfill, "embedding_text_chunks") as m_doc:
                n = self._call(conn, {"asset_id": "a2", "fs_path": str(audio), "modality": "audio"})

        self.assertEqual(n, 1)
        m_plain.assert_called_once()
        m_doc.assert_not_called()  # STT 분기는 문서 로더 미사용
        self.assertEqual(m_plain.call_args.kwargs["embedding_model_name"], _BGE)
        self.assertEqual(cur.executemany.call_args.args[1][0][1], "st_bge")

    def test_missing_document_body_skips_without_raising(self) -> None:
        conn, cur = _mock_conn()
        # 파일 부재 → embedding_text_chunks 가 FileNotFoundError. 백필은 예외를 흡수하고 skip.
        with mock.patch.object(
            self.backfill, "embedding_text_chunks", side_effect=FileNotFoundError("/d/gone.txt")
        ):
            n = self._call(conn, {"asset_id": "a3", "fs_path": "/d/gone.txt", "modality": "txt"})
        self.assertEqual(n, 0)  # 0 = skip
        cur.executemany.assert_not_called()  # INSERT 안 함

    def test_audio_without_sidecar_skips(self) -> None:
        conn, cur = _mock_conn()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "no_stt.wav"
            audio.write_bytes(b"\x00")
            with mock.patch.object(self.backfill, "embedding_plain_text_chunks") as m_plain:
                n = self._call(conn, {"asset_id": "a4", "fs_path": str(audio), "modality": "audio"})
        self.assertEqual(n, 0)
        m_plain.assert_not_called()
        cur.executemany.assert_not_called()

    def test_unreloadable_modality_skips(self) -> None:
        # 시각(image/video) 'st'(VLM 텍스트)·unknown 은 fs_path 에서 본문 재로딩 불가 → skip.
        conn, cur = _mock_conn()
        for modality in ("video", "image", "unknown"):
            with self.subTest(modality=modality):
                n = self._call(conn, {"asset_id": "v", "fs_path": "/d/v.mp4", "modality": modality})
                self.assertEqual(n, 0)
        cur.executemany.assert_not_called()


class TestRunBackfillDbIsolation(unittest.TestCase):
    """배치 격리: 한 자산의 예외가 배치를 멈추지 않고 skip 으로 집계된다(FR-002)."""

    def setUp(self) -> None:
        self.backfill = _load_backfill_module()

    def test_per_asset_exception_is_isolated(self) -> None:
        db = _FakeDB(
            [
                {"asset_id": "a", "fs_path": "/x.txt", "modality": "txt"},
                {"asset_id": "b", "fs_path": "/y.txt", "modality": "txt"},
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
        db = _FakeDB([{"asset_id": "a", "fs_path": "/x.mp4", "modality": "video"}])
        with mock.patch.object(self.backfill, "backfill_asset", return_value=0):
            res = self.backfill.run_backfill_db(
                db, model_name=_BGE, chunk_size=512, encoding="utf-8", normalize=True
            )
        self.assertEqual(res["processed"], 0)
        self.assertEqual(res["skipped"], 1)


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만 실행하는 실 DB e2e (BGE-M3 모델·GPU 필요)")
class TestBackfillBgeE2E(unittest.TestCase):
    """픽스처 st 자산 → 백필 → st·st_bge 공존(chunk 수 일치) + 2회 멱등(중복 0)."""

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
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))
        shutil.rmtree(self._dir, ignore_errors=True)

    def _count(self, asset_id, channel: str) -> int:
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM asset_embedding WHERE asset_id=%s AND channel=%s",
                (asset_id, channel),
            )
            return cur.fetchone()[0]

    def test_backfill_adds_st_bge_alongside_st_idempotently(self) -> None:
        from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
        from src.config.settings import get_current_settings
        from src.database.ids import uuid7
        from src.embedders.text_embedder import _iter_nonempty_chunks

        cfg = get_current_settings()
        # 2~3 청크가 나오도록 chunk_size(512) 보다 긴 한국어 본문.
        txt = self._dir / "doc.txt"
        txt.write_text("가나다라마바사 " * 200, encoding="utf-8")

        # 기존 'st' 채널과 동일한 청킹으로 chunk 수를 산정(모델 없이 더미 벡터로 'st' 행 적재).
        n_chunks = len(
            list(
                _iter_nonempty_chunks(
                    txt, file_kind="txt", encoding=cfg.encoding, chunk_size=cfg.text_embedding_chunk_size
                )
            )
        ) or 1
        aid = uuid7()
        self._ids.append(aid)
        dummy = [0.0] * FIX_EMBEDDING_DIMENSION
        dummy[0] = 0.5
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO asset (asset_id, modality, fs_path, status) VALUES (%s,%s,%s,'registered')",
                (aid, "txt", str(txt)),
            )
            cur.executemany(
                f"INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name) "
                f"VALUES (%s,'st',%s,%s::vector({FIX_EMBEDDING_DIMENSION}),%s)",
                [(aid, i, dummy, cfg.text_embedding_model) for i in range(n_chunks)],
            )

        self.assertEqual(self._count(aid, "st"), n_chunks)
        self.assertEqual(self._count(aid, "st_bge"), 0)

        row = {"asset_id": aid, "fs_path": str(txt), "modality": "txt"}
        # 1회차 백필(실 BGE 임베딩).
        with self.db.transaction() as conn:
            n1 = self.backfill.backfill_asset(
                conn,
                row,
                model_name=cfg.text_embedding_model_bge,
                chunk_size=cfg.text_embedding_chunk_size,
                encoding=cfg.encoding,
                normalize=cfg.text_embedding_normalize,
            )
        self.assertEqual(n1, n_chunks)
        self.assertEqual(self._count(aid, "st"), n_chunks)        # st 무변경
        self.assertEqual(self._count(aid, "st_bge"), n_chunks)    # st_bge 공존, chunk 수 일치

        # 2회차 백필 → ON CONFLICT DO NOTHING 으로 중복 0(멱등, SC-001).
        with self.db.transaction() as conn:
            self.backfill.backfill_asset(
                conn,
                row,
                model_name=cfg.text_embedding_model_bge,
                chunk_size=cfg.text_embedding_chunk_size,
                encoding=cfg.encoding,
                normalize=cfg.text_embedding_normalize,
            )
        self.assertEqual(self._count(aid, "st_bge"), n_chunks)    # 여전히 n_chunks (중복 없음)


if __name__ == "__main__":
    unittest.main()
