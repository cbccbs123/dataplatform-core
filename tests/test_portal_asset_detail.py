"""포탈 자산 상세 조회(``fetch_asset_detail``) mock conn 단위 테스트 (DB 불필요).

검증 의도 (plan 010 D-3)
    - FR-004: ``core_meta``/``ext_meta`` 를 **구분**해 반환(병합하지 않음).
    - FR-005: 임베딩은 채널별 청크 **개수만**(``embedding_channels=[{channel,chunk_count}]``),
      원시 벡터(VECTOR 1536) 미노출. 집계 SQL 이 ``COUNT(*)``·``GROUP BY/ORDER BY channel`` 인지도 검사.
    - FR-006: 관계는 ``graph_query.fetch_active_relations_for_asset`` 결과(양방향) — 주입/모킹.
    - FR-014 노출 게이트: 행 없음 / ``status!='registered'`` / ``domain_label='medical'`` → ``None``.
    - 헌법 3조: 동일 입력 2회 동일 출력.

mock conn 패턴은 test_graph_query / relation_type_catalog 와 동형(``cursor(row_factory=dict_row)``
컨텍스트매니저). ``fetch_active_relations_for_asset`` 는 asset_detail 네임스페이스에서 patch 한다.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _conn_for_detail(asset_row, channel_rows):
    """``conn.cursor(row_factory=dict_row)`` 컨텍스트매니저를 흉내내는 mock conn.

    같은 cur 가 두 번 쓰인다: ① asset+metadata 조회는 ``fetchone`` ② 임베딩 채널 집계는
    ``fetchall``. 둘은 서로 다른 메서드라 한 cur 에 모두 세팅해도 충돌하지 않는다.
    ``execute`` 인자는 call_args_list 로 캡처해 SQL 검사에 쓴다.
    """
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchone.return_value = asset_row
    cur.fetchall.return_value = channel_rows
    conn.cursor.return_value = cur
    return conn, cur


_REGISTERED_ROW = {
    "asset_id": "A1",
    "modality": "text",
    "domain_label": "general",
    "status": "registered",
    "core_meta": {"title": "보고서"},
    "ext_meta": {"pages": 12},
    "tags": ["report", "2026"],
}

_RELATIONS = [
    {
        "asset_id": "B2", "kind_code": "duplicate_near", "is_symmetric": True,
        "direction": "undirected", "confidence": 0.9, "status": "active",
        "topic": {"topic_ko": "사진"}, "reason": "유사", "edge_id": "e1",
    },
]


class TestFetchAssetDetail(unittest.TestCase):
    @patch("src.portal.asset_detail.fetch_active_relations_for_asset")
    def test_happy_path_separates_core_and_ext_meta(self, mock_rel) -> None:
        # FR-004: core_meta/ext_meta 가 별개 키로 반환(병합 금지).
        mock_rel.return_value = list(_RELATIONS)
        conn, _ = _conn_for_detail(
            dict(_REGISTERED_ROW),
            [{"channel": "image_clip", "chunk_count": 3}, {"channel": "text", "chunk_count": 5}],
        )
        from src.portal.asset_detail import fetch_asset_detail

        out = fetch_asset_detail(conn, asset_id="A1")
        self.assertIsNotNone(out)
        self.assertEqual(out["asset_id"], "A1")
        self.assertEqual(out["modality"], "text")
        self.assertEqual(out["domain_label"], "general")
        self.assertEqual(out["status"], "registered")
        self.assertEqual(out["core_meta"], {"title": "보고서"})
        self.assertEqual(out["ext_meta"], {"pages": 12})
        self.assertEqual(out["tags"], ["report", "2026"])

    @patch("src.portal.asset_detail.fetch_active_relations_for_asset")
    def test_embedding_channels_count_only_no_raw_vector(self, mock_rel) -> None:
        # FR-005: 채널별 청크 개수만, 각 dict 은 channel/chunk_count 키만(원시 벡터 없음).
        mock_rel.return_value = []
        conn, _ = _conn_for_detail(
            dict(_REGISTERED_ROW),
            [{"channel": "image_clip", "chunk_count": 3}, {"channel": "text", "chunk_count": 5}],
        )
        from src.portal.asset_detail import fetch_asset_detail

        out = fetch_asset_detail(conn, asset_id="A1")
        self.assertEqual(
            out["embedding_channels"],
            [{"channel": "image_clip", "chunk_count": 3}, {"channel": "text", "chunk_count": 5}],
        )
        for ch in out["embedding_channels"]:
            self.assertEqual(set(ch.keys()), {"channel", "chunk_count"})

    @patch("src.portal.asset_detail.fetch_active_relations_for_asset")
    def test_embedding_query_is_count_aggregate_not_raw_select(self, mock_rel) -> None:
        # 집계 SQL 이 COUNT(*)·GROUP BY/ORDER BY channel 인지(원시 벡터 SELECT 금지, FR-005·헌법 6조).
        mock_rel.return_value = []
        conn, cur = _conn_for_detail(dict(_REGISTERED_ROW), [{"channel": "text", "chunk_count": 2}])
        from src.portal.asset_detail import fetch_asset_detail

        fetch_asset_detail(conn, asset_id="A1")
        sqls = [" ".join(c[0][0].split()) for c in cur.execute.call_args_list]
        agg = next(s for s in sqls if "asset_embedding" in s)
        self.assertIn("COUNT(*)", agg)
        self.assertIn("GROUP BY channel", agg)
        self.assertIn("ORDER BY channel", agg)
        # 원시 벡터 컬럼을 직접 SELECT 하지 않는다.
        self.assertNotIn("SELECT embedding", agg)

    @patch("src.portal.asset_detail.fetch_active_relations_for_asset")
    def test_relations_from_graph_query_seam(self, mock_rel) -> None:
        # FR-006: relations 는 graph_query 결과 그대로, asset_id 키워드로 호출.
        mock_rel.return_value = list(_RELATIONS)
        conn, _ = _conn_for_detail(dict(_REGISTERED_ROW), [])
        from src.portal.asset_detail import fetch_asset_detail

        out = fetch_asset_detail(conn, asset_id="A1")
        self.assertEqual(out["relations"], _RELATIONS)
        mock_rel.assert_called_once_with(conn, asset_id="A1")

    @patch("src.portal.asset_detail.fetch_active_relations_for_asset")
    def test_row_not_found_returns_none(self, mock_rel) -> None:
        # 행 없음 → None(API 404).
        mock_rel.return_value = []
        conn, _ = _conn_for_detail(None, [])
        from src.portal.asset_detail import fetch_asset_detail

        self.assertIsNone(fetch_asset_detail(conn, asset_id="ZZ"))

    @patch("src.portal.asset_detail.fetch_active_relations_for_asset")
    def test_non_registered_returns_none(self, mock_rel) -> None:
        # status != 'registered'(failed/deferred) → None(FR-014/노출 게이트).
        mock_rel.return_value = []
        row = dict(_REGISTERED_ROW)
        row["status"] = "failed"
        conn, _ = _conn_for_detail(row, [])
        from src.portal.asset_detail import fetch_asset_detail

        self.assertIsNone(fetch_asset_detail(conn, asset_id="A1"))

    @patch("src.portal.asset_detail.fetch_active_relations_for_asset")
    def test_medical_returns_none(self, mock_rel) -> None:
        # 의료 자산 배제(FR-014) → None.
        mock_rel.return_value = []
        row = dict(_REGISTERED_ROW)
        row["domain_label"] = "medical"
        conn, _ = _conn_for_detail(row, [])
        from src.portal.asset_detail import fetch_asset_detail

        self.assertIsNone(fetch_asset_detail(conn, asset_id="A1"))

    @patch("src.portal.asset_detail.fetch_active_relations_for_asset")
    def test_determinism_same_input_same_output(self, mock_rel) -> None:
        # 헌법 3조: 동일 입력 2회 동일 출력.
        mock_rel.return_value = list(_RELATIONS)
        from src.portal.asset_detail import fetch_asset_detail

        conn1, _ = _conn_for_detail(dict(_REGISTERED_ROW), [{"channel": "text", "chunk_count": 5}])
        conn2, _ = _conn_for_detail(dict(_REGISTERED_ROW), [{"channel": "text", "chunk_count": 5}])
        self.assertEqual(
            fetch_asset_detail(conn1, asset_id="A1"),
            fetch_asset_detail(conn2, asset_id="A1"),
        )


if __name__ == "__main__":
    unittest.main()
