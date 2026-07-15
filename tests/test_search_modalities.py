"""검색 모달리티 단일 출처(T301·D5·P2-29) 단위 테스트 — DB·네트워크 없음.

리뷰 P2-29: 유효 모달리티 튜플과 CSV 파싱이 3진입점(portal_api·run_search·sample_search_api)에
복제돼 있었다. 이 테스트는 (a) 공유 파서 ``parse_modalities_csv`` 의 순수 파싱 계약,
(b) 유효값 튜플이 코드에 **1벌만** 존재(3진입점이 공유 상수 참조), (c) sample 진입점의
미지 모달리티 응답 형태(200 + {"error"}) 보존을 봉인한다.
"""
from __future__ import annotations

import pathlib
import re
import unittest

from fastapi.testclient import TestClient

from src.config.search_modalities import VALID_SEARCH_MODALITIES, parse_modalities_csv


class TestParseModalitiesCsv(unittest.TestCase):
    """공유 파서 ``parse_modalities_csv`` — split/strip·미지정=None·검증/소문자화 없음(RED ①)."""

    def test_none_and_blank_return_none(self) -> None:
        # 미지정/공백 → None(전체 버킷). 3진입점 공통 계약.
        self.assertIsNone(parse_modalities_csv(None))
        self.assertIsNone(parse_modalities_csv(""))
        self.assertIsNone(parse_modalities_csv("   "))

    def test_comma_split_and_strip(self) -> None:
        # 콤마 분리 + 공백 트림 + 빈 토큰 스킵(기존 3진입점 동일 동작 보존).
        self.assertEqual(parse_modalities_csv("text,image"), ["text", "image"])
        self.assertEqual(parse_modalities_csv(" text , , image "), ["text", "image"])

    def test_case_preserved_not_lowercased(self) -> None:
        # 동작 불변(US-D 원칙): 원본 3파서가 소문자화하지 않았으므로 여기서도 하지 않는다.
        # 대문자/혼합 입력은 원문 그대로 유지 → 이후 유효값(소문자 튜플) 밖으로 거부된다.
        self.assertEqual(parse_modalities_csv("TEXT,Image,VIDEO"), ["TEXT", "Image", "VIDEO"])

    def test_all_blank_returns_none(self) -> None:
        # 콤마만/공백만 → None(빈 리스트가 아니라 None = 전체 버킷).
        self.assertIsNone(parse_modalities_csv(" , , "))


class TestValidModalitiesSingleSource(unittest.TestCase):
    """유효값 튜플이 코드에 1벌만 — 3진입점이 공유 상수를 참조(RED ②)."""

    def test_valid_tuple_value(self) -> None:
        self.assertEqual(VALID_SEARCH_MODALITIES, ("text", "image", "video", "audio"))

    def test_three_entrypoints_reference_shared_constant(self) -> None:
        # 3진입점 모듈이 자체 튜플을 복제하지 않고 공유 상수를 참조(객체 동일성).
        import src.app.portal_api as portal
        import src.app.run_search as run_search
        import src.app.sample_search_api as sample

        self.assertIs(portal.VALID_SEARCH_MODALITIES, VALID_SEARCH_MODALITIES)
        self.assertIs(run_search.VALID_SEARCH_MODALITIES, VALID_SEARCH_MODALITIES)
        self.assertIs(sample.VALID_SEARCH_MODALITIES, VALID_SEARCH_MODALITIES)

    def test_valid_tuple_literal_defined_once(self) -> None:
        # grep 계약: `= ("text","image","video","audio")` 튜플 리터럴이 src/ 전체에서 단 1곳만.
        src_root = pathlib.Path(__file__).resolve().parents[1] / "src"
        pattern = re.compile(
            r'=\s*\(\s*"text"\s*,\s*"image"\s*,\s*"video"\s*,\s*"audio"\s*,?\s*\)'
        )
        hits = sorted(
            p.name
            for p in src_root.rglob("*.py")
            if pattern.search(p.read_text(encoding="utf-8"))
        )
        self.assertEqual(hits, ["search_modalities.py"])


class TestSampleSearchModalityContract(unittest.TestCase):
    """sample_search_api 미지 모달리티 응답 형태(200 + {"error"}) 보존(RED ④)."""

    def test_unknown_modality_returns_200_with_error(self) -> None:
        # sample 은 HTTPException 400 통일(CR-13=US-F) 전이라 **현행 그대로** 200+{"error"} 반환.
        # 미지 모달리티는 search_hybrid 호출 전에 반환되므로 DB/OS 없이 검증된다.
        from src.app.sample_search_api import app

        client = TestClient(app)
        resp = client.get("/search", params={"q": "x", "modalities": "bogus"})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("error", resp.json())
        self.assertIn("bogus", resp.json()["error"])


if __name__ == "__main__":
    unittest.main()
