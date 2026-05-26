"""F-1.3 등록·적재(asset_persist) + 그룹화(grouping) 단위 테스트.

psycopg Connection 을 mock 으로 대체해 SQL 배선·파라미터를 검증(실 DB 불필요).
식별자는 UUIDv7(앱 생성)이므로 uuid7 을 patch 해 결정적으로 검증한다.
실제 DB 적재는 T1-6 통합 수직 슬라이스에서 확인.
"""

from __future__ import annotations

import unittest
import uuid
from unittest import mock

from src.dispatch.types import AssetRecord, EmbeddingItem
from src.registry import asset_persist, grouping
from src.registry.asset_persist import create_asset, finalize_asset
from src.registry.grouping import resolve_group_id, upsert_group

_FIXED = uuid.UUID("018f0000-0000-7000-8000-000000000001")
_GID = uuid.UUID("018f0000-0000-7000-8000-000000000010")


def _mock_conn(fetchone_value):
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = fetchone_value
    return conn, cur


def _executes(cur):
    return cur.execute.call_args_list


class TestCreateAsset(unittest.TestCase):
    def test_generates_uuid7_and_inserts_received(self) -> None:
        conn, cur = _mock_conn(None)
        with mock.patch.object(asset_persist, "uuid7", return_value=_FIXED):
            aid = create_asset(conn, fs_path="/d/a.txt", modality="txt")
        self.assertEqual(aid, _FIXED)
        ins = [c for c in _executes(cur) if "INSERT INTO asset " in c.args[0]]
        self.assertEqual(len(ins), 1)
        # params: (asset_id, group_id, modality, fs_path, file_hash, file_size, domain)
        self.assertEqual(ins[0].args[1], (_FIXED, None, "txt", "/d/a.txt", None, None, "general"))
        self.assertIn("'received'", ins[0].args[0])


class TestFinalizeAsset(unittest.TestCase):
    def test_inserts_metadata_embeddings_and_registers(self) -> None:
        conn, cur = _mock_conn({"status": "extracting"})
        rec = AssetRecord(
            core_meta={"width": 10},
            ext_meta={"summary": "s"},
            tags=["t"],
            fts_plain="hello",
            embeddings=[EmbeddingItem(channel="st", vector=[0.1, 0.2], model_name="legacy")],
        )
        finalize_asset(conn, _FIXED, rec)
        meta_ins = [c for c in _executes(cur) if "INSERT INTO asset_metadata" in c.args[0]]
        self.assertEqual(len(meta_ins), 1)
        self.assertEqual(meta_ins[0].args[1][0], _FIXED)
        self.assertEqual(meta_ins[0].args[1][3], ["t"])
        self.assertEqual(cur.executemany.call_count, 1)
        emb_rows = cur.executemany.call_args.args[1]
        self.assertEqual(emb_rows, [(_FIXED, "st", 0, [0.1, 0.2], "legacy", None)])
        upd = [c for c in _executes(cur) if "UPDATE asset SET status" in c.args[0]]
        self.assertEqual(len(upd), 1)
        self.assertEqual(upd[0].args[1], ("registered", None, _FIXED))

    def test_no_embeddings_skips_executemany(self) -> None:
        conn, cur = _mock_conn({"status": "extracting"})
        finalize_asset(conn, _FIXED, AssetRecord(fts_plain="x"))
        self.assertEqual(cur.executemany.call_count, 0)
        upd = [c for c in _executes(cur) if "UPDATE asset SET status" in c.args[0]]
        self.assertEqual(upd[0].args[1], ("registered", None, _FIXED))


class TestGrouping(unittest.TestCase):
    def test_upsert_group_returns_uuid(self) -> None:
        conn, cur = _mock_conn({"group_id": _GID})
        with mock.patch.object(grouping, "uuid7", return_value=_FIXED):
            gid = upsert_group(conn, group_kind="study_uid", group_key="1.2.3")
        self.assertEqual(gid, _GID)  # RETURNING 값(기존/신규)
        ins = [c for c in _executes(cur) if "INSERT INTO asset_group" in c.args[0]]
        self.assertEqual(ins[0].args[1], (_FIXED, "study_uid", "1.2.3"))

    def test_resolve_group_id_from_study_uid(self) -> None:
        conn, _ = _mock_conn({"group_id": _GID})
        self.assertEqual(resolve_group_id(conn, {"study_uid": "1.2.3"}), _GID)

    def test_resolve_group_id_from_mrn(self) -> None:
        conn, _ = _mock_conn({"group_id": _GID})
        self.assertEqual(resolve_group_id(conn, {"mrn": "M1"}), _GID)

    def test_resolve_group_id_none_when_no_key(self) -> None:
        conn, cur = _mock_conn({"group_id": _GID})
        self.assertIsNone(resolve_group_id(conn, {"width": 10}))
        self.assertEqual(cur.execute.call_count, 0)


if __name__ == "__main__":
    unittest.main()
