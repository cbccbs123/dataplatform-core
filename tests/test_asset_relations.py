"""T3-1 관계 재배선 단위 테스트 (asset_candidates / asset_relation_persist).

psycopg Connection 은 mock 으로 대체해 SQL 배선·필터 로직만 검증한다(실 DB·LLM 불필요).
UUID 타깃 검증, 후보 집합 밖 제외, 자기참조 제외, relation_type 미해결 스킵을 확인한다.
"""

from __future__ import annotations

import unittest
import uuid
from unittest import mock

from src.relations.asset_candidates import _channels_param, find_embedding_candidates
from src.relations.asset_relation_persist import _as_uuid_str, sync_asset_relation_edges

_SRC = "018f0000-0000-7000-8000-000000000001"
_T1 = "018f0000-0000-7000-8000-000000000007"
_T2 = "018f0000-0000-7000-8000-000000000008"


def _mock_conn(rows):
    """fetchall 이 ``rows`` 를 반환하는 mock Connection(dict_row 커서)."""
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = rows
    return conn, cur


class TestChannelsParam(unittest.TestCase):
    def test_st_clip_both(self) -> None:
        self.assertEqual(_channels_param("st"), ["st"])
        self.assertEqual(_channels_param("clip"), ["clip"])
        self.assertEqual(set(_channels_param("both")), {"st", "clip"})


class TestFindCandidates(unittest.TestCase):
    def test_maps_rows_to_str_id_candidates(self) -> None:
        rows = [
            {"id": uuid.UUID(_T1), "file_uri": "/d/a.png", "media_type": "image",
             "emb_score": 0.91, "summary": "요약A"},
            {"id": uuid.UUID(_T2), "file_uri": "/d/b.txt", "media_type": "txt",
             "emb_score": 0.42, "summary": None},
        ]
        conn, cur = _mock_conn(rows)
        out = find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5, embedding_kind="both")
        self.assertEqual([c["id"] for c in out], [_T1, _T2])
        self.assertTrue(all(isinstance(c["id"], str) for c in out))
        self.assertEqual(out[0]["media_type"], "image")
        self.assertEqual(out[1]["summary"], "")  # None → ''
        # SQL 파라미터: (source, channels, source, top_k)
        params = cur.execute.call_args.args[1]
        self.assertEqual(params[0], _SRC)
        self.assertEqual(set(params[1]), {"st", "clip"})
        self.assertEqual(params[3], 5)


class TestAsUuidStr(unittest.TestCase):
    def test_valid_and_invalid(self) -> None:
        self.assertEqual(_as_uuid_str(_T1), _T1)
        self.assertEqual(_as_uuid_str(uuid.UUID(_T1)), _T1)
        self.assertIsNone(_as_uuid_str("not-a-uuid"))
        self.assertIsNone(_as_uuid_str(123))
        self.assertIsNone(_as_uuid_str(None))


class TestSyncAssetRelationEdges(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = mock.MagicMock()
        self.cur = self.conn.cursor.return_value.__enter__.return_value
        self.allowed = frozenset({_T1, _T2})
        self.rtid = uuid.UUID("018f0000-0000-7000-8000-000000000018")

    def _edge(self, **kw):
        e = {"target_media_item_id": _T1, "relation_type_code": "same_domain", "reason": "유사"}
        e.update(kw)
        return e

    def _run(self, edges):
        with mock.patch(
            "src.relations.asset_relation_persist.fetch_relation_type_id_for_normalized_edge",
            return_value=self.rtid,
        ):
            return sync_asset_relation_edges(
                self.conn, source_asset_id=_SRC, edges=edges, allowed_target_ids=self.allowed
            )

    def test_valid_edge_upserted(self) -> None:
        up, sk = self._run([self._edge()])
        self.assertEqual((up, sk), (1, 0))
        sql, params = self.cur.execute.call_args.args[0], self.cur.execute.call_args.args[1]
        self.assertIn("INSERT INTO asset_relation", sql)
        # params: (relation_id, source, target, rtid, conf, reason)
        self.assertEqual(params[1], _SRC)
        self.assertEqual(params[2], _T1)
        self.assertEqual(str(params[0]), params[0])  # relation_id 는 str(uuid7)
        uuid.UUID(params[0])  # 유효 UUID

    def test_target_not_in_allowed_skipped(self) -> None:
        outsider = "018f0000-0000-7000-8000-000000000014"
        up, sk = self._run([self._edge(target_media_item_id=outsider)])
        self.assertEqual((up, sk), (0, 1))
        self.cur.execute.assert_not_called()

    def test_invalid_uuid_target_skipped(self) -> None:
        up, sk = self._run([self._edge(target_media_item_id="42")])
        self.assertEqual((up, sk), (0, 1))

    def test_self_reference_skipped(self) -> None:
        # 소스가 allowed 에 있어도 자기참조는 제외.
        up, sk = self._run([self._edge(target_media_item_id=_SRC)])
        self.assertEqual((up, sk), (0, 1))

    def test_missing_relation_code_skipped(self) -> None:
        up, sk = self._run([self._edge(relation_type_code="")])
        self.assertEqual((up, sk), (0, 1))

    def test_unresolved_relation_type_skipped(self) -> None:
        with mock.patch(
            "src.relations.asset_relation_persist.fetch_relation_type_id_for_normalized_edge",
            return_value=None,
        ):
            up, sk = sync_asset_relation_edges(
                self.conn, source_asset_id=_SRC, edges=[self._edge()],
                allowed_target_ids=self.allowed,
            )
        self.assertEqual((up, sk), (0, 1))


if __name__ == "__main__":
    unittest.main()
