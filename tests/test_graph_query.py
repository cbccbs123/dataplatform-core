"""그래프 read seam ``graph_query`` 단위 테스트 (mock conn, DB 불필요).

검증 의도
    - 대칭 엣지는 캐논 1행으로만 저장되므로(graph_persist._canonical_pair),
      "자산 X의 이웃"을 순진하게 ``WHERE src_node=X`` 로만 찾으면 X가 dst로 접힌
      대칭 엣지를 누락한다(ADR 2026-05-28). 그래서 SQL이 ``sn.asset_id OR dn.asset_id``
      양방향 매칭 + ``relation_kind`` 조인 + status 바인딩을 갖는지(T001) 검증한다.
    - 질의 자산 관점 정규화(대칭→undirected·반대편, 비대칭 src→outbound·dst→inbound)와
      반환 dict 키 세트가 계약대로인지(T002) 검증한다.
    - 결정성(헌법 3조): ``ORDER BY confidence DESC NULLS LAST, edge_id`` 2차 정렬 키 존재.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock


def _conn_returning(rows: list[dict]):
    """``conn.cursor(row_factory=dict_row)`` 컨텍스트매니저를 흉내내는 mock conn.

    relation_type_catalog/review 단위 테스트와 동형 패턴: ``__enter__`` 가 cur 를 돌려주고
    ``fetchall`` 이 주입한 행을 반환한다. ``execute`` 인자는 call_args 로 캡처해 SQL·바인딩 검증.
    """
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchall.return_value = rows
    conn.cursor.return_value = cur
    return conn, cur


class TestGraphQuerySQL(unittest.TestCase):
    """T001 — 실행 SQL이 양방향·relation_kind 조인·status 바인딩·결정적 정렬을 갖는가."""

    def test_sql_matches_both_directions_and_joins_relation_kind(self) -> None:
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, cur = _conn_returning([])
        fetch_active_relations_for_asset(conn, asset_id="A")

        sql = cur.execute.call_args[0][0]
        compact = " ".join(sql.split())  # 줄바꿈·들여쓰기 정규화로 견고한 부분문자열 검사

        # ① 대칭 엣지 누락 방지를 위한 양방향 매칭(src/dst 어느 쪽이든 X면 매칭)
        self.assertIn("sn.asset_id = %s OR dn.asset_id = %s", compact)
        # ② is_symmetric·kind_code 는 relation_kind 에만 있으므로 반드시 조인
        self.assertIn("JOIN relation_kind", compact)
        # 양 끝점 asset 해소를 위한 node 역조인(asset 노드만)
        self.assertIn("node_kind = 'asset'", compact)

    def test_status_is_bound_parameter_not_hardcoded(self) -> None:
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, cur = _conn_returning([])
        fetch_active_relations_for_asset(conn, asset_id="A", status="proposed")

        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        compact = " ".join(sql.split())
        # ③ status 는 바인딩(%s) — 호출자가 active/proposed 등 선택 가능
        self.assertIn("ge.status = %s", compact)
        # 양방향 asset_id 2개 + status 1개 = (X, X, status) 순서
        self.assertEqual(params, ("A", "A", "proposed"))

    def test_order_by_has_edge_id_secondary_for_determinism(self) -> None:
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, cur = _conn_returning([])
        fetch_active_relations_for_asset(conn, asset_id="A")

        sql = cur.execute.call_args[0][0]
        compact = " ".join(sql.split())
        # ④ confidence 동점 시 순서 불안정 방지를 위한 edge_id 2차 정렬(헌법 3조, plan R-3)
        self.assertIn("ORDER BY ge.confidence DESC NULLS LAST, ge.edge_id", compact)


class TestGraphQueryNormalize(unittest.TestCase):
    """T002 — 질의 자산 관점 정규화 분기와 반환 dict 키 세트."""

    _EXPECTED_KEYS = {
        "asset_id", "kind_code", "is_symmetric", "direction",
        "confidence", "status", "topic", "reason", "edge_id",
        # 057 FR-102: 이웃 표시필드 하향(하위호환 필드 추가)
        "file_name", "modality",
    }

    def _row(self, **over):
        """SQL(dict_row) 한 행을 흉내. graph_query 가 select 하는 컬럼명 그대로.

        057 FR-102: node→asset 조인으로 양끝 자산의 modality·fs_path 를 함께 끌어온다.
        """
        base = {
            "edge_id": "e1",
            "kind_code": "duplicate_near",
            "is_symmetric": True,
            "confidence": 0.95,
            "reason": "유사",
            "topic": {"topic_ko": "사진"},
            "status": "active",
            "src_asset": "A",
            "dst_asset": "B",
            "src_modality": "text",
            "dst_modality": "image",
            "src_fs_path": "/data/raw/문서A.txt",
            "dst_fs_path": "/data/raw/사진B.png",
        }
        base.update(over)
        return base

    def test_symmetric_edge_is_undirected_and_returns_other_side(self) -> None:
        # 대칭 kind: 방향은 무방향, 이웃 asset_id 는 질의 자산의 반대편
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, _ = _conn_returning([self._row(is_symmetric=True, src_asset="A", dst_asset="B")])
        out = fetch_active_relations_for_asset(conn, asset_id="A")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["direction"], "undirected")
        self.assertEqual(out[0]["asset_id"], "B")  # 반대편

    def test_symmetric_edge_query_from_dst_side_still_returns_other(self) -> None:
        # X 가 dst 로 접힌 대칭 엣지(ADR 핵심 시나리오): 이웃은 src 쪽이어야 함
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, _ = _conn_returning([self._row(is_symmetric=True, src_asset="A", dst_asset="B")])
        out = fetch_active_relations_for_asset(conn, asset_id="B")
        self.assertEqual(out[0]["direction"], "undirected")
        self.assertEqual(out[0]["asset_id"], "A")  # 질의 자산(B)의 반대편 = A

    def test_asymmetric_src_is_outbound(self) -> None:
        # 비대칭이며 질의 자산이 src → outbound, 이웃은 dst
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, _ = _conn_returning([self._row(is_symmetric=False, src_asset="A", dst_asset="B")])
        out = fetch_active_relations_for_asset(conn, asset_id="A")
        self.assertEqual(out[0]["direction"], "outbound")
        self.assertEqual(out[0]["asset_id"], "B")

    def test_asymmetric_dst_is_inbound(self) -> None:
        # 비대칭이며 질의 자산이 dst → inbound, 이웃은 src
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, _ = _conn_returning([self._row(is_symmetric=False, src_asset="A", dst_asset="B")])
        out = fetch_active_relations_for_asset(conn, asset_id="B")
        self.assertEqual(out[0]["direction"], "inbound")
        self.assertEqual(out[0]["asset_id"], "A")

    def test_return_dict_has_exact_contract_keys(self) -> None:
        # 반환 dict 키 세트가 계약(plan D-1)과 정확히 일치 — 소비자(상세·묶음) 일관성
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, _ = _conn_returning([self._row()])
        out = fetch_active_relations_for_asset(conn, asset_id="A")
        self.assertEqual(set(out[0].keys()), self._EXPECTED_KEYS)
        # 통과 필드는 원본 보존
        self.assertEqual(out[0]["kind_code"], "duplicate_near")
        self.assertEqual(out[0]["confidence"], 0.95)
        self.assertEqual(out[0]["topic"], {"topic_ko": "사진"})
        self.assertEqual(out[0]["edge_id"], "e1")

    def test_neighbor_carries_other_side_file_name_and_modality(self) -> None:
        # FR-102(057): 이웃 dict 에 상대 자산의 file_name(fs_path basename)·modality 를 싣는다.
        # 질의 자산 A(src) → 이웃은 dst(B): dst 쪽 modality/fs_path 를 취해야 한다.
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, _ = _conn_returning([self._row(is_symmetric=True, src_asset="A", dst_asset="B")])
        out = fetch_active_relations_for_asset(conn, asset_id="A")
        self.assertEqual(out[0]["asset_id"], "B")
        self.assertEqual(out[0]["file_name"], "사진B.png")  # dst basename
        self.assertEqual(out[0]["modality"], "image")       # dst modality

    def test_neighbor_from_dst_side_takes_src_file_name_and_modality(self) -> None:
        # 질의 자산이 dst 인 경우(대칭 엣지가 접힘): 이웃은 src → src 쪽 modality/fs_path.
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, _ = _conn_returning([self._row(is_symmetric=True, src_asset="A", dst_asset="B")])
        out = fetch_active_relations_for_asset(conn, asset_id="B")
        self.assertEqual(out[0]["asset_id"], "A")
        self.assertEqual(out[0]["file_name"], "문서A.txt")  # src basename
        self.assertEqual(out[0]["modality"], "text")        # src modality

    def test_sql_joins_asset_for_both_endpoints(self) -> None:
        # FR-102: modality·fs_path 는 asset 에만 있으므로 양끝 node→asset 조인 필요(assets_in_topic 패턴).
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, cur = _conn_returning([])
        fetch_active_relations_for_asset(conn, asset_id="A")
        compact = " ".join(cur.execute.call_args[0][0].split())
        self.assertIn("sa.modality", compact)
        self.assertIn("da.modality", compact)
        self.assertIn("sa.fs_path", compact)
        self.assertIn("da.fs_path", compact)

    def test_determinism_same_input_same_output(self) -> None:
        # 같은 입력 2회 → 같은 결과(헌법 3조). 정규화는 순수하므로 안정적.
        from src.relations.graph_query import fetch_active_relations_for_asset

        rows = [
            self._row(edge_id="e1", confidence=0.9, src_asset="A", dst_asset="B"),
            self._row(edge_id="e2", confidence=0.9, src_asset="C", dst_asset="A", is_symmetric=False),
        ]
        conn1, _ = _conn_returning([dict(r) for r in rows])
        conn2, _ = _conn_returning([dict(r) for r in rows])
        self.assertEqual(
            fetch_active_relations_for_asset(conn1, asset_id="A"),
            fetch_active_relations_for_asset(conn2, asset_id="A"),
        )


if __name__ == "__main__":
    unittest.main()
