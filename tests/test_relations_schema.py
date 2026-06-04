"""relations: schema 파서·검증·리졸버 단위 테스트."""

from __future__ import annotations

import unittest

from src.relations.prompt import build_relation_proposal_prompt
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


# ── [008 그룹4] T013·T014·T015: 경로 패턴 힌트 프롬프트(레버 A) ─────────────
class TestRelationProposalPromptPathSignals(unittest.TestCase):
    """T013·T014 [US3, FR-008·FR-009, SC-006] — 프롬프트 빌더의 경로 신호 노출·가드.

    - T013(FR-008): 후보 JSON 에 디렉터리 풀경로를 노출하지 않고 **basename 만** 노출(헌법 3조·10조).
    - T014(FR-009): 파일명·폴더 패턴 가이드 + "내용 합치 시에만" 가드 문구. C-3: 경로 신호
      후보(emb_score=0.0)를 LLM 이 '비유사'로 오해하지 않게 "경로 신호" 표식/문구를 둔다.
    """

    _ABS_DIR = "/data/secret_patient_folder/2025"
    _BASENAME = "report_summary.txt"
    _FULL = f"{_ABS_DIR}/{_BASENAME}"

    def _build(self, *, emb_score: float = 0.0) -> str:
        return build_relation_proposal_prompt(
            source_summary="소스 요약",
            source_media_type="txt",
            candidates=[
                {
                    "id": "018f0000-0000-7000-8000-000000000007",
                    "file_uri": self._FULL,
                    "media_type": "txt",
                    "emb_score": emb_score,
                    "summary": "후보 요약",
                }
            ],
            relation_kinds_catalog=[
                {"type_code": "same_series", "type_name": "연작", "description": "연작"},
                {"type_code": "derived_from", "type_name": "파생", "description": "파생"},
                {"type_code": "references", "type_name": "참조", "description": "참조"},
            ],
        )

    def test_basename_present(self) -> None:
        # FR-008/SC-006: 후보의 basename 은 프롬프트에 포함돼 LLM 이 파일명 패턴을 본다.
        prompt = self._build()
        self.assertIn(self._BASENAME, prompt)

    def test_directory_fullpath_absent(self) -> None:
        # FR-008/SC-006: 디렉터리 풀경로(절대경로)는 단 한 건도 노출되지 않는다(PHI·결정성).
        prompt = self._build()
        self.assertNotIn(self._ABS_DIR, prompt)
        self.assertNotIn(self._FULL, prompt)
        # 'file_uri' 키 자체로 풀경로를 내보내지 않는다(basename 전용 키로 치환).
        self.assertNotIn(self._ABS_DIR, prompt)

    def test_guard_phrase_present(self) -> None:
        # FR-009: "내용이 합치할 때만 same_series/derived_from/references 를 고르라" 가드 문구.
        prompt = self._build()
        self.assertIn("내용", prompt)
        self.assertIn("same_series", prompt)
        self.assertIn("derived_from", prompt)
        self.assertIn("references", prompt)

    def test_path_pattern_hints_present(self) -> None:
        # FR-009: 1부/2부 연작·요약/번역/전사 파생·참조 패턴 힌트가 가이드에 포함.
        prompt = self._build()
        self.assertIn("파일명", prompt)
        self.assertIn("폴더", prompt)

    def test_path_signal_marker_for_zero_emb_score(self) -> None:
        # C-3: emb_score=0.0 인 경로 신호 후보를 LLM 이 '비유사'로 오해하지 않게
        # "경로 신호" 표식 문구가 프롬프트에 포함된다.
        prompt = self._build(emb_score=0.0)
        self.assertIn("경로 신호", prompt)


if __name__ == "__main__":
    unittest.main()
