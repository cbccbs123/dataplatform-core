"""049 G1 — VLM 요약 키워드 후처리 순수 코어 단위 테스트(FR-401/402·SC-003).

``src.llm.summary_postprocess`` 의 순수 함수(정규화·objects 승격)를 LLM·DB·settings 없이 검증한다.
헌법 3조(결정성): 동일 입력 2회 호출 → 동일 리스트. set 은 dedup 전용이고 출력 순서는 first-seen
보존(정렬 아님)이라, BM25 무영향이면서 VLM relevance 순서를 살린다. generic 일반어("영상" 등)는
검색 신호 0이라 제거한다. 후처리는 순수 Python·LLM 0(FR-402)이라 mock 불필요.
"""

from __future__ import annotations

import unittest

from src.llm.summary_postprocess import normalize_keywords, promote_objects_to_keywords


class TestImportSkeleton(unittest.TestCase):
    """T101: 모듈·공개 시그니처 존재(스켈레톤 단계 — import 가능 + 함수 호출 가능)."""

    def test_module_exports_callables(self) -> None:
        self.assertTrue(callable(normalize_keywords))
        self.assertTrue(callable(promote_objects_to_keywords))


class TestNormalizeKeywords(unittest.TestCase):
    """T102: 공백 trim·casefold dedup·generic 제거·first-seen 순서(FR-401·헌법 3조)."""

    def test_normalize(self) -> None:
        # "충전기"/" 충전기 " 는 strip 후 동일(dedup), "영상" 은 generic 제거, 순서는 등장 순(정렬 아님).
        self.assertEqual(
            normalize_keywords(["충전기", " 충전기 ", "영상", "USB"]),
            ["충전기", "USB"],
        )


class TestPromoteObjects(unittest.TestCase):
    """T103: 키워드 정규화 후 objects 를 후보로 합치되 중복·generic 제외·limit cap(FR-402)."""

    def test_promote_objects(self) -> None:
        # "충전기" 는 이미 키워드라 중복 제외, "케이블" 은 승격, "영상" 은 generic 제거.
        self.assertEqual(
            promote_objects_to_keywords(["충전기"], ["충전기", "케이블", "영상"], limit=10),
            ["충전기", "케이블"],
        )

    def test_limit_caps_total(self) -> None:
        # limit 은 키워드+승격 합산 상한. 키워드가 이미 limit 이면 objects 미승격.
        self.assertEqual(
            promote_objects_to_keywords(["a", "b"], ["c", "d"], limit=2),
            ["a", "b"],
        )
        # objects 로 limit 까지만 채운다.
        self.assertEqual(
            promote_objects_to_keywords(["a"], ["b", "c", "d"], limit=3),
            ["a", "b", "c"],
        )


class TestDeterminismAndEmpty(unittest.TestCase):
    """T104: 헌법 3조 결정성(동일 입력 2회 → 동일 리스트) + 빈 입력 경계."""

    def test_determinism(self) -> None:
        kws = ["충전기", "USB", "충전기", "케이블"]
        objs = ["어댑터", "케이블", "영상"]
        # 동일 입력 2회 호출 → 동일 리스트(set dedup 이 순서를 흔들지 않음을 봉인).
        self.assertEqual(normalize_keywords(kws), normalize_keywords(kws))
        self.assertEqual(
            promote_objects_to_keywords(kws, objs, limit=10),
            promote_objects_to_keywords(kws, objs, limit=10),
        )

    def test_empty(self) -> None:
        self.assertEqual(normalize_keywords([]), [])
        self.assertEqual(promote_objects_to_keywords([], [], limit=5), [])
        # None 입력도 안전(`keywords or []` 폴백).
        self.assertEqual(normalize_keywords(None), [])
        self.assertEqual(promote_objects_to_keywords(None, None, limit=5), [])


if __name__ == "__main__":
    unittest.main()
