"""프롬프트 변형 seam — 기본값이 **채택된 운영 문구**(2026-07-29·shadow A/B 검증)임을 봉인한다.

원래 이 봉인은 "seam 추출 전과 바이트 동일"을 지켰다. B팔 문구가 검증을 거쳐 운영 기본값으로
채택된 뒤로는 **새 문구가 기본값**임을 지킨다 — 특히 옛 순환 지시("임베딩 유사도로 가져온
후보처럼")가 되돌아오면 여기서 막는다(전 후보가 임베딩 유사도로 온 것이라 그 힌트는
"모든 후보=duplicate_near" 지시가 된다 · docs/관계_품질_측정_20260728.md §4).
"""
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


class TestPromptAdoptedDefaults(unittest.TestCase):
    def test_채택된_힌트_문구가_기본값이다(self):
        p = build_relation_proposal_prompt(**_KW)
        self.assertIn("같은 구체적 대상", p)          # duplicate_near 힌트
        self.assertIn("다루는 대상이 다르면", p)      # anti-dup 구분 문구
        self.assertIn("대상은 다르지만 같은 분야", p)  # same_domain 힌트

    def test_옛_순환_지시가_되돌아오지_않는다(self):
        # "임베딩 유사도로 가져온 후보처럼"은 전 후보에 duplicate_near 를 붙이라는 순환 지시였다.
        p = build_relation_proposal_prompt(**_KW)
        self.assertNotIn("임베딩 유사도로 가져온 후보처럼", p)
        self.assertNotIn("유사도·근접 후보라면", p)

    def test_override를_주지_않으면_출력이_동일하다(self):
        self.assertEqual(
            build_relation_proposal_prompt(**_KW),
            build_relation_proposal_prompt(**_KW, kind_hints_override=None,
                                           anti_dup_override=None))


class TestPromptVariantInjection(unittest.TestCase):
    # 주입 마커는 기본 문구와 부분 문자열조차 겹치지 않게 고른다 — 겹치면 "갈아끼웠다"는
    # 검증이 기본값만으로도 통과해 버린다.
    def test_kind_힌트를_갈아끼울_수_있다(self):
        p = build_relation_proposal_prompt(
            **_KW, kind_hints_override={"duplicate_near": "주입된-테스트-힌트-마커"})
        self.assertIn("주입된-테스트-힌트-마커", p)
        self.assertNotIn("같은 구체적 대상", p)   # 기본 duplicate_near 힌트가 교체됐다

    def test_anti_dup_문구를_갈아끼울_수_있다(self):
        p = build_relation_proposal_prompt(
            **_KW, anti_dup_override="\n\n**구분:** 주입된-구분-마커.")
        self.assertIn("주입된-구분-마커", p)
        self.assertNotIn("다루는 대상이 다르면", p)   # 기본 구분 문구가 교체됐다

    def test_갈아끼우지_않은_kind는_원래_힌트를_유지한다(self):
        p = build_relation_proposal_prompt(
            **_KW, kind_hints_override={"duplicate_near": "새 문구"})
        self.assertIn("대상은 다르지만 같은 분야", p)   # same_domain 은 그대로


class TestAntiDupOverrideBeatsCatalog(unittest.TestCase):
    """**명시 주입은 카탈로그 조건을 이긴다** — 081 Y팔 측정이 요구한 성질(2026-07-30).

    자동 부착은 ``duplicate_near`` ∧ ``same_domain`` 동시 활성일 때만 일어난다. 그 조건에
    override 까지 종속시켰던 탓에 **"종류를 빼고 억제 문구는 살린다"는 실험 자체가 불가능**했다
    (X팔에서 종류 제거와 문구 소실이 겹쳐 dup 이 24.2%→72.2% 로 튀었고, 원인을 분리할 수 없었다).
    """

    _DUP_ONLY = [{"type_code": "duplicate_near", "description": "d"}]
    _KW_DUP_ONLY = {**_KW, "relation_kinds_catalog": _DUP_ONLY}

    def test_same_domain_없는_카탈로그에도_주입한_문구는_붙는다(self):
        p = build_relation_proposal_prompt(
            **self._KW_DUP_ONLY, anti_dup_override="\n\n**구분:** 주입된-구분-마커.")
        self.assertIn("주입된-구분-마커", p)

    def test_override_없으면_한쪽만_있는_카탈로그엔_붙지_않는다(self):
        # 운영 동작 봉인 — 자동 부착 조건은 그대로다(override 를 줄 때만 조건을 넘는다).
        p = build_relation_proposal_prompt(**self._KW_DUP_ONLY)
        self.assertNotIn("다루는 대상이 다르면", p)

    def test_빈문자열_override는_문구를_제거한다(self):
        # ``""`` 는 "미지정"이 아니라 **"문구를 넣지 말라"는 명시**다. truthy 검사로 바꾸면
        # 빈 문자열이 조용히 무시되고 기본 문구가 되살아난다 — 문구 소실만 재현하는 실험이 깨진다.
        p = build_relation_proposal_prompt(**_KW, anti_dup_override="")
        self.assertNotIn("다루는 대상이 다르면", p)
        self.assertIn("``duplicate_near``:", p)   # 가이드 본문은 남는다


if __name__ == "__main__":
    unittest.main()
