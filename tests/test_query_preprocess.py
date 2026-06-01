"""006 검색 — 질의 구조화 폴백(client 주입, 네트워크 없음).

``structure_user_query`` 는 LLM 응답을 구조화 dict 로 만든다. LLM 이 스키마를 어겨
비-dict JSON(배열·스칼라)을 내도 예외 없이 안전한 기본 dict 로 폴백해야 한다(#6).
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.search.query_preprocess import structure_user_query


def _client_returning(content: str) -> MagicMock:
    """``complete_text`` 가 호출하는 ``chat.completions.create`` 응답을 모킹."""
    c = MagicMock()
    c.chat.completions.create.return_value.choices = [MagicMock()]
    c.chat.completions.create.return_value.choices[0].message.content = content
    return c


class TestStructureUserQueryFallback(unittest.TestCase):
    def test_valid_json_object_parsed(self) -> None:
        c = _client_returning('{"keywords": ["a"], "semantic_query": "요약"}')
        out = structure_user_query("워크숍 발표자료", client=c)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["keywords"], ["a"])
        self.assertEqual(out["semantic_query"], "요약")

    def test_json_array_falls_back_to_dict(self) -> None:
        c = _client_returning("[1, 2, 3]")
        out = structure_user_query("워크숍 발표자료", client=c)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["semantic_query"], "워크숍 발표자료")

    def test_json_scalar_falls_back_to_dict(self) -> None:
        c = _client_returning("42")
        out = structure_user_query("워크숍 발표자료", client=c)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["semantic_query"], "워크숍 발표자료")

    def test_empty_response_falls_back_to_dict(self) -> None:
        c = _client_returning("")
        out = structure_user_query("워크숍 발표자료", client=c)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["semantic_query"], "워크숍 발표자료")


if __name__ == "__main__":
    unittest.main()
