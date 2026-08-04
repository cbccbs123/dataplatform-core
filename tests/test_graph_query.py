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
        # 081: 상태 다중 조회(2단 노출)로 ANY 바인딩이 됐다 — 하드코딩이 아니라는 의도는 그대로.
        self.assertIn("ge.status = ANY(%s)", compact)
        # 양방향 asset_id 2개 + status 1개 = (X, X, status) 순서
        # 081: 상태를 목록으로 바인딩한다(2단 노출은 active+proposed 를 함께 읽는다).
        self.assertEqual(params, ("A", "A", ["proposed"]))

    def test_order_by_has_edge_id_secondary_for_determinism(self) -> None:
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, cur = _conn_returning([])
        fetch_active_relations_for_asset(conn, asset_id="A")

        sql = cur.execute.call_args[0][0]
        compact = " ".join(sql.split())
        # ④ confidence 동점 시 순서 불안정 방지를 위한 edge_id 2차 정렬(헌법 3조, plan R-3)
        self.assertIn("ORDER BY ge.confidence DESC NULLS LAST, ge.edge_id", compact)

    def test_no_domain_exclusion(self) -> None:
        # 2026-07-23: 도메인 제외 전면 제거 — 관계 조회 SQL 이 medical 을 배제하지 않는다(의료 복귀 시 재도입).
        from src.relations.graph_query import fetch_active_relations_for_asset

        conn, cur = _conn_returning([])
        fetch_active_relations_for_asset(conn, asset_id="A")
        compact = " ".join(cur.execute.call_args[0][0].split())
        self.assertNotIn("medical", compact)


class TestGraphQueryNormalize(unittest.TestCase):
    """T002 — 질의 자산 관점 정규화 분기와 반환 dict 키 세트."""

    _EXPECTED_KEYS = {
        "asset_id", "kind_code", "is_symmetric", "direction",
        "confidence", "status", "topic", "reason", "edge_id",
        # 057 FR-102: 이웃 표시필드 하향(하위호환 필드 추가)
        "file_name", "modality",
        # 081 조각③: 노출 등급(strong/weak) — 역시 하위호환 필드 추가다(기존 키 불변).
        "tier",
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

# ── 081 노출 2단 ────────────────────────────────────────────────────────────────
def _row(*, status="active", kind="same_domain", conf=0.9, edge="e1",
         src="a1", dst="a2"):
    """노출 등급 테스트용 최소 행(기존 _conn_returning 이 그대로 먹는 모양)."""
    return {"edge_id": edge, "kind_code": kind, "is_symmetric": True,
            "confidence": conf, "reason": None, "topic": None, "status": status,
            "src_asset": src, "dst_asset": dst,
            "src_modality": "text", "src_fs_path": "/d/a1.txt",
            "dst_modality": "text", "dst_fs_path": "/d/a2.txt"}


class TestExposureTiers(unittest.TestCase):
    """081 조각③ — 강칸(연관 자료)·약칸(비슷한 주제) 2단 노출.

    약칸이 필요한 이유: 자동승인 게이트가 `same_domain` 을 강등하면 관계 보유 자산이 26% 줄어
    화면이 빈다. 강등분을 약칸으로 살려 커버리지를 지키면서 강칸 정밀도만 올린다.
    """

    def test_active_행에_strong_등급이_붙는다(self):
        from src.relations.graph_query import fetch_relations_for_asset
        conn, _ = _conn_returning([_row(status="active")])
        rows = fetch_relations_for_asset(conn, asset_id="a1")
        self.assertEqual([r["tier"] for r in rows], ["strong"])

    def test_include_weak_이면_proposed_고신뢰도_함께_온다(self):
        from src.relations.graph_query import fetch_relations_for_asset
        conn, _ = _conn_returning([_row(status="proposed", conf=0.9)])
        rows = fetch_relations_for_asset(conn, asset_id="a1",
                                         include_weak=True, min_conf_similarity=0.75)
        self.assertEqual([r["tier"] for r in rows], ["weak"])

    def test_include_weak_이_아니면_proposed_를_조회하지_않는다(self):
        # 상태 바인딩 자체가 active 뿐이어야 한다 — 읽어와서 버리면 쓸데없이 무겁다.
        from src.relations.graph_query import fetch_relations_for_asset
        conn, cur = _conn_returning([])
        fetch_relations_for_asset(conn, asset_id="a1", include_weak=False)
        params = cur.execute.call_args[0][1]
        self.assertEqual(params[2], ["active"])

    def test_include_weak_이면_두_상태를_바인딩한다(self):
        from src.relations.graph_query import fetch_relations_for_asset
        conn, cur = _conn_returning([])
        fetch_relations_for_asset(conn, asset_id="a1", include_weak=True)
        self.assertEqual(cur.execute.call_args[0][1][2], ["active", "proposed"])

    def test_저신뢰_proposed_는_include_weak_에도_안_온다(self):
        from src.relations.graph_query import fetch_relations_for_asset
        conn, _ = _conn_returning([_row(status="proposed", conf=0.6)])
        rows = fetch_relations_for_asset(conn, asset_id="a1",
                                         include_weak=True, min_conf_similarity=0.75)
        self.assertEqual(rows, [])

    def test_명시적_계열_저신뢰_proposed_는_약칸으로_온다(self):
        from src.relations.graph_query import fetch_relations_for_asset
        conn, _ = _conn_returning([_row(status="proposed", kind="references", conf=0.2)])
        rows = fetch_relations_for_asset(conn, asset_id="a1",
                                         include_weak=True, min_conf_similarity=0.75)
        self.assertEqual([r["tier"] for r in rows], ["weak"])

    def test_강칸이_약칸보다_먼저_온다(self):
        # 신뢰도만으로 정렬하면 고신뢰 약칸이 저신뢰 강칸을 밀어낸다.
        from src.relations.graph_query import fetch_relations_for_asset
        conn, _ = _conn_returning([
            _row(status="proposed", conf=0.99, edge="e-weak", dst="a3"),
            _row(status="active", conf=0.80, edge="e-strong", dst="a2")])
        rows = fetch_relations_for_asset(conn, asset_id="a1",
                                         include_weak=True, min_conf_similarity=0.75)
        self.assertEqual([r["tier"] for r in rows], ["strong", "weak"])

    def test_같은_등급_안에서는_DB_정렬을_보존한다(self):
        # 신뢰도·edge_id 정렬은 SQL 이 한다(mock 은 ORDER BY 를 적용하지 않으므로 여기서
        # 검증할 수 없다 — SQL 쪽은 TestGraphQuerySQL 이 본다). 여기서 봉인하는 것은
        # **등급 정렬이 안정 정렬이라 DB 순서를 흩뜨리지 않는다**는 점이다.
        from src.relations.graph_query import fetch_relations_for_asset
        conn, _ = _conn_returning([
            _row(status="active", conf=0.9, edge="e-hi", dst="a2"),
            _row(status="active", conf=0.7, edge="e-lo", dst="a3")])
        rows = fetch_relations_for_asset(conn, asset_id="a1")
        self.assertEqual([r["edge_id"] for r in rows], ["e-hi", "e-lo"])

    def test_등급_정렬이_섞인_입력에서도_안정적이다(self):
        # 약칸이 앞에 와도 강칸이 올라오되, 같은 등급 내부 순서는 입력(=DB) 순서를 지킨다.
        from src.relations.graph_query import fetch_relations_for_asset
        conn, _ = _conn_returning([
            _row(status="proposed", conf=0.99, edge="w1", dst="a3"),
            _row(status="active", conf=0.90, edge="s1", dst="a2"),
            _row(status="proposed", conf=0.80, edge="w2", dst="a4"),
            _row(status="active", conf=0.75, edge="s2", dst="a5")])
        rows = fetch_relations_for_asset(conn, asset_id="a1",
                                         include_weak=True, min_conf_similarity=0.75)
        self.assertEqual([r["edge_id"] for r in rows], ["s1", "s2", "w1", "w2"])

    def test_질의_자산_관점_정규화가_유지된다(self):
        # 기존 seam 의 핵심 계약 — 이웃은 늘 반대편이고 대칭이면 undirected.
        from src.relations.graph_query import fetch_relations_for_asset
        conn, _ = _conn_returning([_row(status="active", src="a2", dst="a1")])
        rows = fetch_relations_for_asset(conn, asset_id="a1")
        self.assertEqual(rows[0]["asset_id"], "a2")
        self.assertEqual(rows[0]["direction"], "undirected")


class TestBackwardCompatibleWrapper(unittest.TestCase):
    """기존 함수의 계약 불변 — 포탈 상세·다운로드 번들이 이 키들을 쓴다."""

    def test_기존_함수가_그대로_동작한다(self):
        from src.relations.graph_query import fetch_active_relations_for_asset
        conn, _ = _conn_returning([_row(status="active")])
        rows = fetch_active_relations_for_asset(conn, asset_id="a1")
        for key in ("asset_id", "kind_code", "is_symmetric", "direction", "confidence",
                    "status", "topic", "reason", "edge_id", "file_name", "modality"):
            self.assertIn(key, rows[0], f"기존 키 {key} 가 사라졌다")

    def test_기존_함수는_status_인자를_그대로_받는다(self):
        from src.relations.graph_query import fetch_active_relations_for_asset
        conn, cur = _conn_returning([])
        fetch_active_relations_for_asset(conn, asset_id="a1", status="rejected")
        self.assertEqual(cur.execute.call_args[0][1][2], ["rejected"])

