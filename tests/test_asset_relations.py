"""T3-1 관계 후보 검색 단위 테스트 (asset_candidates). DB·LLM 불필요.

엣지 영속화는 단계 C에서 graph_edge 로 이관됨(graph_persist) — tests/test_graph_persist.py 참조.
"""
from __future__ import annotations

import unittest
import uuid
from unittest import mock

from src.relations.asset_candidates import _channels_param, find_embedding_candidates
from src.relations.llm_propose import parse_and_normalize_edges

_SRC = "018f0000-0000-7000-8000-000000000001"
_T1 = "018f0000-0000-7000-8000-000000000007"
_T2 = "018f0000-0000-7000-8000-000000000008"


def _edge(confidence: object) -> dict:
    """confidence 값만 바꿔 가며 검증할 최소 LLM 엣지 dict."""
    return {
        "target_media_item_id": _T1,
        "relation_type_code": "same_series",
        "confidence": confidence,
        "reason": "테스트",
    }


class TestConfidenceClamp(unittest.TestCase):
    """T004 [US4, FR-010, #2] — confidence 를 [0,1] 로 클램프하고, 비정상값은 결정적 0.0."""

    def _conf(self, raw: object) -> float:
        out = parse_and_normalize_edges({"edges": [_edge(raw)]})
        self.assertEqual(len(out), 1)
        return out[0]["confidence"]

    def test_above_one_clamped_to_one(self) -> None:
        # 1.5 → 1.0: 자동승인 임계 판정이 1.0 을 넘지 않도록 상한 클램프.
        self.assertEqual(self._conf(1.5), 1.0)

    def test_below_zero_clamped_to_zero(self) -> None:
        # -0.3 → 0.0: 음수 confidence 를 하한 0.0 으로 클램프.
        self.assertEqual(self._conf(-0.3), 0.0)

    def test_in_range_preserved(self) -> None:
        # 정상 범위 값은 그대로 보존.
        self.assertEqual(self._conf(0.42), 0.42)

    def test_nan_falls_back_to_zero(self) -> None:
        # NaN → 0.0: 비교 불가능한 값은 결정적 기본값으로(헌법 3조).
        self.assertEqual(self._conf(float("nan")), 0.0)

    def test_unparsable_string_falls_back_to_zero(self) -> None:
        # 파싱 불가 문자열 → 0.0.
        self.assertEqual(self._conf("높음"), 0.0)

    def test_missing_falls_back_to_zero(self) -> None:
        # confidence 키 누락 → 0.0.
        edge = _edge(0.5)
        del edge["confidence"]
        out = parse_and_normalize_edges({"edges": [edge]})
        self.assertEqual(out[0]["confidence"], 0.0)

    def test_numeric_string_in_range_parsed(self) -> None:
        # 숫자 문자열도 float 로 파싱되어 보존.
        self.assertEqual(self._conf("0.8"), 0.8)


def _mock_conn(rows):
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
            {"id": uuid.UUID(_T1), "file_uri": "/d/a.png", "media_type": "image", "emb_score": 0.91, "summary": "요약A"},
            {"id": uuid.UUID(_T2), "file_uri": "/d/b.txt", "media_type": "txt", "emb_score": 0.42, "summary": None},
        ]
        conn, cur = _mock_conn(rows)
        out = find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5, embedding_kind="both")
        self.assertEqual([c["id"] for c in out], [_T1, _T2])
        self.assertTrue(all(isinstance(c["id"], str) for c in out))
        self.assertEqual(out[1]["summary"], "")  # None → ''
        params = cur.execute.call_args.args[1]
        self.assertEqual(params[0], _SRC)
        self.assertEqual(set(params[1]), {"st", "clip"})
        self.assertEqual(params[3], 0.0)  # min_sim 기본값
        self.assertEqual(params[4], 5)    # top_k


class TestDeterministicOrdering(unittest.TestCase):
    """T003 [US2, FR-007] — best_sim 동률 시 후보 id ASC tiebreaker 로 결정적 정렬(헌법 3조)."""

    def test_order_by_has_id_tiebreaker(self) -> None:
        # ORDER BY 가 best_sim DESC 만이면 동률 후보 순서가 비결정적 — id ASC 보조 정렬 필수.
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        sql = cur.execute.call_args.args[0]
        # 정규화: 공백을 단일화해 줄바꿈·들여쓰기에 무관하게 부분 문자열 검사.
        norm = " ".join(sql.split())
        self.assertIn("ORDER BY p.best_sim DESC, p.id ASC", norm)

    def test_id_tiebreaker_is_after_best_sim(self) -> None:
        # best_sim 가 1순위, id 가 2순위여야 유사도 우선순위가 보존된다.
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        norm = " ".join(cur.execute.call_args.args[0].split())
        order_clause = norm[norm.index("ORDER BY"):]
        self.assertLess(order_clause.index("best_sim"), order_clause.index("p.id"))


if __name__ == "__main__":
    unittest.main()
