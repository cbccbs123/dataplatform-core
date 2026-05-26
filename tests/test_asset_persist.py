"""F-1.3 등록·적재(asset_persist) + 그룹화(grouping) 단위 테스트.

psycopg Connection 을 mock 으로 대체해 SQL 배선·파라미터를 검증(실 DB 불필요).
실제 DB 적재는 T1-6 통합 수직 슬라이스에서 확인.
"""

from __future__ import annotations

import unittest
from unittest import mock

from src.dispatch.types import AssetRecord, EmbeddingItem
from src.registry.asset_persist import create_asset, finalize_asset
from src.registry.grouping import resolve_group_id, upsert_group


def _mock_conn(fetchone_value):
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = fetchone_value
    return conn, cur


def _executes(cur):
    return cur.execute.call_args_list


class TestCreateAsset(unittest.TestCase):
    def test_returns_asset_id_and_inserts_received(self) -> None:
        conn, cur = _mock_conn({"asset_id": 42})
        aid = create_asset(conn, fs_path="/d/a.txt", modality="txt")
        self.assertEqual(aid, 42)
        ins = [c for c in _executes(cur) if "INSERT INTO asset " in c.args[0]]
        self.assertEqual(len(ins), 1)
        # params: (group_id, modality, fs_path, file_hash, file_size, domain)
        self.assertEqual(ins[0].args[1], (None, "txt", "/d/a.txt", None, None, "general"))
        self.assertIn("'received'", ins[0].args[0])

    def test_no_returning_raises(self) -> None:
        conn, _ = _mock_conn(None)
        with self.assertRaises(RuntimeError):
            create_asset(conn, fs_path="/d/a.txt", modality="txt")


class TestFinalizeAsset(unittest.TestCase):
    def test_inserts_metadata_embeddings_and_registers(self) -> None:
        # finalize 마지막 set_status 가 extracting→registered 를 검증하므로 현재 상태 extracting.
        conn, cur = _mock_conn({"status": "extracting"})
        rec = AssetRecord(
            core_meta={"width": 10},
            ext_meta={"summary": "s"},
            tags=["t"],
            fts_plain="hello",
            embeddings=[EmbeddingItem(channel="st", vector=[0.1, 0.2], model_name="legacy")],
        )
        finalize_asset(conn, 7, rec)
        meta_ins = [c for c in _executes(cur) if "INSERT INTO asset_metadata" in c.args[0]]
        self.assertEqual(len(meta_ins), 1)
        # core/ext 는 json 문자열로, tags 는 리스트로 전달
        self.assertEqual(meta_ins[0].args[1][0], 7)
        self.assertEqual(meta_ins[0].args[1][3], ["t"])
        # 임베딩 executemany 1회
        self.assertEqual(cur.executemany.call_count, 1)
        emb_rows = cur.executemany.call_args.args[1]
        self.assertEqual(emb_rows, [(7, "st", 0, [0.1, 0.2], "legacy", None)])
        # 상태 registered 로 UPDATE
        upd = [c for c in _executes(cur) if "UPDATE asset SET status" in c.args[0]]
        self.assertEqual(len(upd), 1)
        self.assertEqual(upd[0].args[1], ("registered", None, 7))

    def test_no_embeddings_skips_executemany(self) -> None:
        conn, cur = _mock_conn({"status": "extracting"})
        finalize_asset(conn, 8, AssetRecord(fts_plain="x"))
        self.assertEqual(cur.executemany.call_count, 0)
        upd = [c for c in _executes(cur) if "UPDATE asset SET status" in c.args[0]]
        self.assertEqual(upd[0].args[1], ("registered", None, 8))


class TestGrouping(unittest.TestCase):
    def test_upsert_group_returns_id(self) -> None:
        conn, cur = _mock_conn({"group_id": 5})
        self.assertEqual(upsert_group(conn, group_kind="study_uid", group_key="1.2.3"), 5)

    def test_resolve_group_id_from_study_uid(self) -> None:
        conn, _ = _mock_conn({"group_id": 9})
        self.assertEqual(resolve_group_id(conn, {"study_uid": "1.2.3"}), 9)

    def test_resolve_group_id_from_mrn(self) -> None:
        conn, _ = _mock_conn({"group_id": 11})
        self.assertEqual(resolve_group_id(conn, {"mrn": "M1"}), 11)

    def test_resolve_group_id_none_when_no_key(self) -> None:
        conn, cur = _mock_conn({"group_id": 1})
        self.assertIsNone(resolve_group_id(conn, {"width": 10}))
        # 키 없으면 DB 호출 없음
        self.assertEqual(cur.execute.call_count, 0)


if __name__ == "__main__":
    unittest.main()
