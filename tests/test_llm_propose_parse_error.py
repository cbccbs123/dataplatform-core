"""069 T005(P1-3) — 관계 LLM 파싱실패를 예외 승격(pending 재시도)·정상 빈 제안과 구분. LLM 0.

isolated 3갈래 구분(069 재확인 2026-07-14):
  ① 정상 빈 제안({"edges": []}) → 통과 → edges 0·error None → isolated **유지**(옳음)
  ② 066 무내용 스킵(asset_entry 조기 반환·LLM 호출 전) → isolated **유지**(의도된 경로·본 수정 무접점)
  ③ 파싱실패({}·스키마 불능) → RelationProposalParseError → run_relations except → **pending**(신규)
①은 본 파일, ②는 066 테스트, ③의 pending 전이는 test_run_relations_retry(exception→pending)가 커버.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import src.relations.llm_propose as lp


def _with_llm(out):
    return patch("src.llm.client.complete_json", return_value=out)


class TestProposeEdgesJsonParseError(unittest.TestCase):
    def test_empty_dict_raises_for_retry(self) -> None:
        # ③ client 파싱실패 폴백 {} → 예외 승격(조용한 isolated 오확정 차단).
        with _with_llm({}):
            with self.assertRaises(lp.RelationProposalParseError):
                lp.propose_edges_json("p")

    def test_unrecognizable_schema_raises(self) -> None:
        # ③ 엣지 구조가 전혀 없는 응답(스키마 불능)도 재시도 대상.
        with _with_llm({"answer": "관계 없음"}):
            with self.assertRaises(lp.RelationProposalParseError):
                lp.propose_edges_json("p")

    def test_normal_empty_edges_passes(self) -> None:
        # ① 정상 빈 제안 — 예외 아님·그대로 반환(→ 하류에서 isolated 유지·옳은 종결).
        with _with_llm({"edges": []}):
            self.assertEqual(lp.propose_edges_json("p"), {"edges": []})

    def test_items_alias_and_single_edge_pass(self) -> None:
        # 기존 허용 형태(items 별칭·단일 엣지 객체) 불변.
        with _with_llm({"items": [{"target_media_item_id": "u1"}]}):
            self.assertIn("items", lp.propose_edges_json("p"))
        with _with_llm({"target_media_item_id": "u1", "confidence": 0.8}):
            self.assertEqual(lp.propose_edges_json("p")["target_media_item_id"], "u1")

    def test_error_is_exception_for_asset_isolation(self) -> None:
        # run_relations 자산 단위 `except Exception` 이 잡아 pending 으로 보낼 수 있는 타입 계약.
        self.assertTrue(issubclass(lp.RelationProposalParseError, Exception))


if __name__ == "__main__":
    unittest.main()
