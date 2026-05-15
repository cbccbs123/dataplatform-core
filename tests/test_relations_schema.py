"""relations: schema 파서·검증·리졸버 단위 테스트."""

from __future__ import annotations

import unittest

from src.relations.resolve_relation_type import resolve_relation_type_code
from src.relations.schema import (
    coerce_topic_fields_mvp,
    description_ko_from_type_name_ko,
    extract_topic_fields_from_edge,
    normalize_relation_type_code,
    normalize_subtopic_en,
    normalize_subtopic_ko,
    parse_llm_edges,
    sanitize_llm_proposed_type_code,
    target_ref_for_media_item,
    type_label_from_kind_code,
    validate_topic_fields,
)

STRUCTURAL_TYPE_CODES = frozenset({"duplicate_near", "derived_from", "references"})


class TestParseLlmEdges(unittest.TestCase):
    def test_edges_key(self) -> None:
        data = {"edges": [{"target_media_item_id": 1}]}
        self.assertEqual(len(parse_llm_edges(data)), 1)

    def test_items_key(self) -> None:
        data = {"items": [{"target_media_item_id": 2}]}
        self.assertEqual(parse_llm_edges(data)[0]["target_media_item_id"], 2)

    def test_skips_non_dict(self) -> None:
        data = {"edges": [1, {"target_media_item_id": 3}]}
        self.assertEqual(len(parse_llm_edges(data)), 1)


class TestTopicFields(unittest.TestCase):
    def test_extract_aliases(self) -> None:
        edge = {"topic": "결혼", "subtopic": "비용", "topic_en": "wedding", "subtopic_en": "cost"}
        self.assertEqual(
            extract_topic_fields_from_edge(edge),
            ("결혼", "비용", "wedding", "cost"),
        )

    def test_topic_first_ko_segment_and_en_two_tokens(self) -> None:
        edge = {
            "topic_ko": "교통안전",
            "subtopic_ko": "보행자 안전수칙",
            "topic_en": "traffic safety",
            "subtopic_en": "pedestrian safety rules",
        }
        self.assertEqual(
            extract_topic_fields_from_edge(edge),
            ("교통안전", "보행자", "traffic safety", "pedestrian"),
        )

    def test_validate_requires_topic(self) -> None:
        self.assertFalse(validate_topic_fields("", "")[0])
        self.assertTrue(validate_topic_fields("게임", "")[0])

    def test_coerce_from_reason(self) -> None:
        edge = {"reason": "스팀 하드웨어와 게임 추천", "topic_ko": "", "subtopic_ko": ""}
        tk, sk, te, se, auto = coerce_topic_fields_mvp(edge)
        self.assertTrue(auto)
        self.assertIn("스팀", tk)
        self.assertEqual(sk, "")
        self.assertEqual(te, "general")
        self.assertEqual(se, "")

    def test_coerce_default_when_empty(self) -> None:
        edge: dict = {}
        tk, sk, te, se, auto = coerce_topic_fields_mvp(edge)
        self.assertEqual(tk, "일반")
        self.assertTrue(auto)
        self.assertEqual(sk, "")
        self.assertEqual(te, "general")
        self.assertEqual(se, "")

    def test_coerce_keeps_explicit_topic(self) -> None:
        edge = {
            "topic_ko": "교통",
            "subtopic_ko": "안전",
            "topic_en": "traffic",
            "subtopic_en": "safety",
            "reason": "x",
        }
        tk, sk, te, se, auto = coerce_topic_fields_mvp(edge)
        self.assertFalse(auto)
        self.assertEqual(tk, "교통")
        self.assertEqual(sk, "안전")
        self.assertEqual(te, "traffic")
        self.assertEqual(se, "safety")


class TestNormalizeSubtopic(unittest.TestCase):
    def test_ko_first_word_only(self) -> None:
        self.assertEqual(
            normalize_subtopic_ko("  보행자   안전   수칙   추가   문장  "),
            "보행자",
        )

    def test_ko_empty(self) -> None:
        self.assertEqual(normalize_subtopic_ko(""), "")
        self.assertEqual(normalize_subtopic_ko("   "), "")

    def test_en_first_token_only(self) -> None:
        self.assertEqual(
            normalize_subtopic_en("Pedestrian Safety Rules And More"),
            "pedestrian",
        )

    def test_en_strips_edge_punct(self) -> None:
        self.assertEqual(normalize_subtopic_en(" , foo bar , "), "foo")


class TestKindLabelFallback(unittest.TestCase):
    def test_label_from_snake_case(self) -> None:
        self.assertEqual(type_label_from_kind_code("gaming_hardware"), "gaming hardware")

    def test_description_from_spaced_label(self) -> None:
        self.assertEqual(
            description_ko_from_type_name_ko(type_label_from_kind_code("gaming_hardware")),
            "gaming·hardware 도메인 연관",
        )

    def test_single_token_domain_line(self) -> None:
        self.assertEqual(description_ko_from_type_name_ko("의료"), "의료 도메인 연관")

    def test_label_truncates(self) -> None:
        long_code = "gaming_" * 30 + "hardware"
        out = type_label_from_kind_code(long_code, max_len=25)
        self.assertLessEqual(len(out), 25)

    def test_empty_code_label(self) -> None:
        self.assertEqual(type_label_from_kind_code(""), "기타")


class TestNormalizeRelationTypeCode(unittest.TestCase):
    def test_normalize_lowercase(self) -> None:
        self.assertEqual(normalize_relation_type_code("  Medical "), "medical")


class TestSanitizeLlmProposedTypeCode(unittest.TestCase):
    def test_ok_slug(self) -> None:
        self.assertEqual(sanitize_llm_proposed_type_code("gaming_hardware"), "gaming_hardware")

    def test_allows_catalog_like_slug(self) -> None:
        self.assertEqual(sanitize_llm_proposed_type_code("duplicate_near"), "duplicate_near")

    def test_rejects_legacy_domain(self) -> None:
        self.assertIsNone(sanitize_llm_proposed_type_code("computer"))
        self.assertIsNone(sanitize_llm_proposed_type_code("medical"))

    def test_rejects_hyphen(self) -> None:
        self.assertIsNone(sanitize_llm_proposed_type_code("my-type"))

    def test_rejects_uppercase(self) -> None:
        self.assertIsNone(sanitize_llm_proposed_type_code("Gaming"))

    def test_rejects_too_long(self) -> None:
        self.assertIsNone(sanitize_llm_proposed_type_code("a" + "b" * 100))

    def test_max_length_100(self) -> None:
        self.assertEqual(sanitize_llm_proposed_type_code("a" + "b" * 99), "a" + "b" * 99)


class TestResolveRelationType(unittest.TestCase):
    def setUp(self) -> None:
        self.allowed = frozenset(
            {"same_domain", "same_series", "duplicate_near", "references", "derived_from"}
        )

    def test_candidate_with_kind_code(self) -> None:
        self.assertEqual(
            resolve_relation_type_code(
                target_in_candidate_set=True,
                llm_relation_kind_code="same_domain",
                allowed_relation_kind_codes=self.allowed,
            ),
            "same_domain",
        )

    def test_derived_wins_over_kind(self) -> None:
        self.assertEqual(
            resolve_relation_type_code(
                target_in_candidate_set=False,
                derived_path_detected=True,
                llm_relation_kind_code="same_domain",
                allowed_relation_kind_codes=self.allowed,
            ),
            "derived_from",
        )

    def test_citation_wins(self) -> None:
        self.assertEqual(
            resolve_relation_type_code(
                target_in_candidate_set=False,
                citation_detected=True,
                allowed_relation_kind_codes=self.allowed,
            ),
            "references",
        )

    def test_outside_candidate_none_even_with_kind(self) -> None:
        self.assertIsNone(
            resolve_relation_type_code(
                target_in_candidate_set=False,
                llm_relation_kind_code="same_domain",
                allowed_relation_kind_codes=self.allowed,
            )
        )

    def test_not_in_allowed_returns_none(self) -> None:
        self.assertIsNone(
            resolve_relation_type_code(
                target_in_candidate_set=True,
                llm_relation_kind_code="finance",
                allowed_relation_kind_codes=self.allowed,
            )
        )


class TestStructuralCodes(unittest.TestCase):
    def test_expected_structural(self) -> None:
        self.assertIn("duplicate_near", STRUCTURAL_TYPE_CODES)


class TestTargetRef(unittest.TestCase):
    def test_format(self) -> None:
        self.assertEqual(target_ref_for_media_item(42), "id:42")


if __name__ == "__main__":
    unittest.main()
