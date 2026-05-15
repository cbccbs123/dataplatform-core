"""LLM 신규 relation_type 한글 라벨 정규화(네트워크 없음). 폴백은 ``schema.type_label_from_kind_code`` 기준."""

from __future__ import annotations

import unittest

from src.relations.llm_propose import normalize_type_ko_labels_from_llm_response


class TestNormalizeTypeKoLabelsFromLlm(unittest.TestCase):
    def test_valid_response(self) -> None:
        tn, desc = normalize_type_ko_labels_from_llm_response(
            {"type_name_ko": "게이밍 하드웨어", "description_ko": "게이밍·하드웨어 도메인 연관"},
            type_code="gaming_hardware",
        )
        self.assertEqual(tn, "게이밍 하드웨어")
        self.assertEqual(desc, "게이밍·하드웨어 도메인 연관")

    def test_fallback_on_bad_suffix(self) -> None:
        tn, desc = normalize_type_ko_labels_from_llm_response(
            {"type_name_ko": "게이밍 하드웨어", "description_ko": "잘못된 설명"},
            type_code="gaming_hardware",
        )
        self.assertEqual(tn, "gaming hardware")
        self.assertEqual(desc, "gaming·hardware 도메인 연관")

    def test_fallback_on_empty(self) -> None:
        tn, desc = normalize_type_ko_labels_from_llm_response({}, type_code="semiconductor")
        self.assertEqual(tn, "semiconductor")
        self.assertEqual(desc, "semiconductor 도메인 연관")


if __name__ == "__main__":
    unittest.main()
