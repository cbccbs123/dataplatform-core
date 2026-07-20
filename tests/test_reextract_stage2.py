"""050 — stage-2 video 키프레임 VLM 재캡션·재임베딩 모드의 순수/mock 단위.

실 VLM·DB 없이 검증 가능한 부분만:
  - 게이트(FR-101): VLM_SUMMARY_PROMPT_V2=off·--force 없음 → stage-2 진입 중단(return 2)
  - `_reprocess_video_stage2`(P1·P2): video_skill 재사용 → 임베딩 DELETE+INSERT·ext_meta UPDATE
  - --dry-run 쓰기 0 / fs_path 부재 skip / 멱등(SC-005)

실 백필·recall·관계 골든은 G3(사람·코퍼스·RUN_DB_E2E·049 v2 flip).
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import src.dispatch.types as mod_types  # AssetRecord/EmbeddingItem — video_skill mock 반환 구성용

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reextract_summaries.py"
_spec = importlib.util.spec_from_file_location("reextract_summaries", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class _FakeCursor:
    """conn.cursor() 컨텍스트 매니저 mock — execute 호출을 (sql, params) 로 기록."""

    def __init__(self, calls: list[tuple[str, object]]) -> None:
        self._calls = calls

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self._calls.append((sql, params))

    def fetchall(self) -> list:
        return []


class _FakeConn:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.calls)


class Stage2GateTest(unittest.TestCase):
    """FR-101 — v2 off·force 없음이면 stage-2 진입 차단(v1 재캡션 무의미)."""

    def _run_main(self, argv: list[str], *, v2: bool):
        cfg = SimpleNamespace(
            vlm=SimpleNamespace(summary_prompt_v2=v2),
            embed=SimpleNamespace(normalize=True),
        )
        with (
            mock.patch.object(sys, "argv", ["reextract_summaries.py", *argv]),
            mock.patch("dotenv.load_dotenv"),
            mock.patch("src.config.settings.init_settings"),
            mock.patch("src.config.settings.get_current_settings", return_value=cfg),
            mock.patch("src.database.postgres_util.PostgresUtil") as db_cls,
        ):
            rc = _mod.main()
        return rc, db_cls

    def test_stage2_refuses_when_v2_off(self) -> None:
        # v2 off + force 없음 → 게이트가 return 2 로 중단, DB 연결 시도조차 없어야.
        rc, db_cls = self._run_main(["--env", "dev", "--stage2"], v2=False)
        self.assertEqual(rc, 2)
        db_cls.assert_not_called()

    def test_stage2_force_bypasses_gate(self) -> None:
        # --force 면 v2 off 라도 게이트 통과 → DB 경로로 진입(이후는 mock DB).
        rc, db_cls = self._run_main(["--env", "dev", "--stage2", "--force", "--dry-run"], v2=False)
        self.assertNotEqual(rc, 2)


class _DryRunCursor:
    """SELECT 결과를 고정 행으로 돌려주는 dry-run 집계 검증용 커서."""

    def __init__(self, rows: list) -> None:
        self._rows = rows
        self.writes: list = []

    def __enter__(self) -> _DryRunCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        # SELECT 만 허용 — dry-run 은 어떤 write SQL 도 실행하면 안 된다(쓰기 0).
        if not sql.lstrip().upper().startswith("SELECT"):
            self.writes.append((sql, params))

    def fetchall(self) -> list:
        return self._rows


class _DryRunConn:
    def __init__(self, rows: list) -> None:
        self._cursor = _DryRunCursor(rows)

    def cursor(self) -> _DryRunCursor:
        return self._cursor

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class Stage2DryRunTest(unittest.TestCase):
    """T003 — --stage2 --dry-run: video 대상 수·fs_path 부재 집계·쓰기 0."""

    def _run(self, rows: list, *, v2: bool = True):
        cfg = SimpleNamespace(vlm=SimpleNamespace(summary_prompt_v2=v2), embed=SimpleNamespace(normalize=True))
        conn = _DryRunConn(rows)
        db_ctx = mock.MagicMock()
        db_ctx.__enter__.return_value = db_ctx
        db_ctx.connection.return_value.__enter__.return_value = conn
        with (
            mock.patch.object(
                sys,
                "argv",
                ["reextract_summaries.py", "--env", "dev", "--stage2", "--dry-run"],
            ),
            mock.patch("dotenv.load_dotenv"),
            mock.patch("src.config.settings.init_settings"),
            mock.patch("src.config.settings.get_current_settings", return_value=cfg),
            mock.patch("src.database.postgres_util.PostgresUtil", return_value=db_ctx),
        ):
            rc = _mod.main()
        return rc, conn

    def test_dry_run_no_writes_and_video_scope(self) -> None:
        rows = [
            ("v1", "video", "/missing.mp4", {}),
            ("v2", "video", "/missing2.mp4", {}),
            ("t1", "txt", "/doc.txt", {}),
        ]
        with self.assertLogs("meta_extract.reextract", level="INFO") as cm:
            rc, conn = self._run(rows)
        self.assertEqual(rc, 0)
        # dry-run 은 어떤 write(DELETE/INSERT/UPDATE) 도 실행하지 않는다(쓰기 0).
        self.assertEqual(conn._cursor.writes, [])
        joined = "\n".join(cm.output)
        # stage-2 dry-run 은 video 로 한정해 대상 수·fs_path 부재 건을 집계해야 한다.
        self.assertIn("stage-2", joined)
        self.assertIn("video", joined)
        # 두 파일 모두 부재 → 부재 집계가 2 로 보고되어야 한다(txt 는 stage-2 대상 아님).
        self.assertRegex(joined, r"부재[^0-9]*2")


class _RecordingCursor:
    """write SQL 을 (kind, params) 로 기록하는 커서 — DELETE/INSERT/UPDATE 호출 검증용."""

    def __init__(self, log: list) -> None:
        self._log = log

    def __enter__(self) -> _RecordingCursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        head = sql.lstrip().split()[0].upper()
        self._log.append((head, sql, params))


class _RecordingConn:
    def __init__(self) -> None:
        self.log: list = []

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self.log)


def _fake_embedding_items(mod):
    """_embed_video mock 반환 — 키프레임 2개 × (st·clip) = 4 EmbeddingItem."""
    EmbeddingItem = mod.EmbeddingItem
    return [
        EmbeddingItem(channel="st", vector=[0.1, 0.2], model_name="kosim", chunk_index=0),
        EmbeddingItem(channel="clip", vector=[0.3, 0.4], model_name="clip", chunk_index=0),
        EmbeddingItem(channel="st", vector=[0.5, 0.6], model_name="kosim", chunk_index=1),
        EmbeddingItem(channel="clip", vector=[0.7, 0.8], model_name="clip", chunk_index=1),
    ]


class ReprocessVideoStage2Test(unittest.TestCase):
    """T101~T103 — _reprocess_video_stage2 코어(P1 video_skill 재사용·P2 DELETE+INSERT)."""

    def setUp(self) -> None:
        self.cfg = SimpleNamespace(vlm=SimpleNamespace(summary_prompt_v2=True), embed=SimpleNamespace(normalize=True))
        # ext_meta 에 keyframes 2개를 둔 AssetRecord mock(키프레임 수 반환 검증용).
        self.rec = mod_types.AssetRecord(
            core_meta={},
            ext_meta={"keyframes": [{"scene_index": 0}, {"scene_index": 1}], "summary": "v2 요약"},
            tags=[],
            embeddings=[],
        )

    def _patches(self, exists: bool = True):
        items = _fake_embedding_items(mod_types)
        return (
            mock.patch.object(_mod, "_extract_video_meta", return_value=self.rec),
            mock.patch.object(_mod, "_embed_video", return_value=items),
            mock.patch("pathlib.Path.is_file", return_value=exists),
            items,
        )

    def test_replaces_embeddings_and_updates_ext_meta(self) -> None:
        ext_patch, emb_patch, file_patch, items = self._patches()
        conn = _RecordingConn()
        with ext_patch, emb_patch, file_patch:
            out = _mod._reprocess_video_stage2(
                conn, "aid-1", "/v.mp4", cfg=self.cfg, dry_run=False
            )
        # 반환 계약: 키프레임 수·임베딩 수.
        self.assertEqual(out, {"keyframes": 2, "embeddings": len(items)})
        kinds = [k for k, _s, _p in conn.log]
        # ① 자산 임베딩 DELETE 1회 ② items 수만큼 INSERT ③ ext_meta UPDATE 1회.
        self.assertEqual(kinds.count("DELETE"), 1)
        self.assertEqual(kinds.count("INSERT"), len(items))
        self.assertEqual(kinds.count("UPDATE"), 1)
        # DELETE 는 자산 단위(WHERE asset_id) — 채널 한정이 아니라 전 채널 교체여야.
        del_sql = next(s for k, s, _p in conn.log if k == "DELETE")
        self.assertIn("asset_id", del_sql)
        self.assertNotIn("channel", del_sql.lower())
        # ext_meta UPDATE 는 updated_at 도 갱신해야 한다(앱-SET 규약·여타 write 경로와 일관).
        upd_sql = next(s for k, s, _p in conn.log if k == "UPDATE")
        self.assertIn("updated_at", upd_sql.lower())

    def test_dry_run_no_writes(self) -> None:
        # T102 — dry_run=True: extract/embed 는 호출하되 영속(write) 0.
        ext_patch, emb_patch, file_patch, items = self._patches()
        conn = _RecordingConn()
        with ext_patch as ext_m, emb_patch, file_patch:
            out = _mod._reprocess_video_stage2(
                conn, "aid-1", "/v.mp4", cfg=self.cfg, dry_run=True
            )
        self.assertEqual(conn.log, [])  # write 0
        ext_m.assert_called_once()  # extract 는 호출(대상 집계·계약)
        self.assertEqual(out["embeddings"], len(items))

    def test_missing_file_skips(self) -> None:
        # T103 — fs_path 부재 → None(skip)·extract 호출 0·예외 전파 0.
        ext_patch, emb_patch, file_patch, _items = self._patches(exists=False)
        conn = _RecordingConn()
        with ext_patch as ext_m, emb_patch, file_patch:
            out = _mod._reprocess_video_stage2(
                conn, "aid-1", "/missing.mp4", cfg=self.cfg, dry_run=False
            )
        self.assertIsNone(out)
        ext_m.assert_not_called()
        self.assertEqual(conn.log, [])

    def test_idempotent_same_input_same_calls(self) -> None:
        # T103 — 동일 입력 2회(mock 결정적) → 동일 write 호출 시퀀스(SC-005).
        ext_patch, emb_patch, file_patch, _items = self._patches()
        with ext_patch, emb_patch, file_patch:
            conn_a = _RecordingConn()
            _mod._reprocess_video_stage2(conn_a, "aid-1", "/v.mp4", cfg=self.cfg, dry_run=False)
        ext_patch2, emb_patch2, file_patch2, _i2 = self._patches()
        with ext_patch2, emb_patch2, file_patch2:
            conn_b = _RecordingConn()
            _mod._reprocess_video_stage2(conn_b, "aid-1", "/v.mp4", cfg=self.cfg, dry_run=False)
        self.assertEqual(
            [(k, p) for k, _s, p in conn_a.log],
            [(k, p) for k, _s, p in conn_b.log],
        )


if __name__ == "__main__":
    unittest.main()
