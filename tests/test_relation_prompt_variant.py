"""프롬프트 변형 seam — 기본값이 기존 출력과 바이트 동일함을 봉인한다."""
import unittest

from src.relations.prompt import build_relation_proposal_prompt

_CATALOG = [
    {"type_code": "duplicate_near", "description": "d"},
    {"type_code": "same_domain", "description": "s"},
]
_CANDS = [{"id": "a2", "file_uri": "/x/b.txt", "media_type": "text",
           "emb_score": 0.9, "summary": "요약", "topic_ko": "역사", "subtopic_ko": "궁궐"}]
_KW = {"source_summary": "소스 요약", "source_media_type": "text",
       "candidates": _CANDS, "relation_kinds_catalog": _CATALOG}


class TestPromptUnchangedByDefault(unittest.TestCase):
    def test_기존_힌트_문구가_그대로_들어간다(self):
        p = build_relation_proposal_prompt(**_KW)
        self.assertIn("임베딩 유사도로 가져온 후보처럼", p)
        self.assertIn("유사도·근접 후보라면", p)

    def test_override를_주지_않으면_출력이_동일하다(self):
        self.assertEqual(
            build_relation_proposal_prompt(**_KW),
            build_relation_proposal_prompt(**_KW, kind_hints_override=None,
                                           anti_dup_override=None))


class TestPromptVariantInjection(unittest.TestCase):
    def test_kind_힌트를_갈아끼울_수_있다(self):
        p = build_relation_proposal_prompt(
            **_KW, kind_hints_override={"duplicate_near": "사실상 중복본일 때"})
        self.assertIn("사실상 중복본일 때", p)
        self.assertNotIn("임베딩 유사도로 가져온 후보처럼", p)

    def test_anti_dup_문구를_갈아끼울_수_있다(self):
        p = build_relation_proposal_prompt(**_KW, anti_dup_override="\n\n**구분:** 대상이 다르면 아니다.")
        self.assertIn("대상이 다르면 아니다", p)
        self.assertNotIn("유사도·근접 후보라면", p)

    def test_갈아끼우지_않은_kind는_원래_힌트를_유지한다(self):
        p = build_relation_proposal_prompt(
            **_KW, kind_hints_override={"duplicate_near": "새 문구"})
        self.assertIn("둘 다 게임, 둘 다 교통", p)   # same_domain 은 그대로
