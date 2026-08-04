"""골든 초안 검수 도구의 순수 함수 — 표시·조립·표식 제거."""
from __future__ import annotations

import unittest

from scripts.review_golden_draft import build_golden, display_name, render_pair

_INFO = {
    "/d/019f__경복궁 야간개방.txt": {
        "name": "019f__경복궁 야간개방.txt", "modality": "text",
        "topic": "역사·문화유산", "subtopic": "궁궐", "summary": "경복궁 야간개방 안내와 관람 정보"},
    "/d/019f__창덕궁 후원.mp4": {
        "name": "019f__창덕궁 후원.mp4", "modality": "video",
        "topic": "역사·문화유산", "subtopic": "궁궐", "summary": "창덕궁 후원 영상"},
}


class TestDisplayName(unittest.TestCase):
    def test_재수집_명명에서_제목만_남긴다(self):
        # 파일명이 `<uuid>__<제목>` 이라 uuid 를 보여주면 화면이 읽히지 않는다.
        got = display_name("/d/019f__경복궁 야간개방.txt", _INFO)
        self.assertTrue(got.startswith("경복궁 야간개방"))
        self.assertNotIn("019f__", got)

    def test_모달리티를_함께_보여준다(self):
        self.assertIn("(video)", display_name("/d/019f__창덕궁 후원.mp4", _INFO))

    def test_정보가_없으면_경로_끝을_쓴다(self):
        got = display_name("/very/long/path/to/unknown.txt", {})
        self.assertIn("unknown.txt", got)
        self.assertIn("(?)", got)


class TestRenderPair(unittest.TestCase):
    PAIR = {"a": "/d/019f__경복궁 야간개방.txt", "b": "/d/019f__창덕궁 후원.mp4",
            "kind": "duplicate_near"}

    def test_양끝_요약을_함께_보여준다(self):
        # 이름만으로는 "제주도 vs 섬 일반" 같은 개체 수준 구분이 안 된다.
        out = render_pair(1, 69, self.PAIR, _INFO)
        self.assertIn("경복궁 야간개방 안내", out)
        self.assertIn("창덕궁 후원 영상", out)

    def test_진행률과_제안_종류를_보여준다(self):
        out = render_pair(7, 69, self.PAIR, _INFO)
        self.assertIn("[7/69]", out)
        self.assertIn("duplicate_near", out)

    def test_주제와_세부주제를_보여준다(self):
        out = render_pair(1, 69, self.PAIR, _INFO)
        self.assertIn("역사·문화유산>궁궐", out)


class TestBuildGolden(unittest.TestCase):
    def test_초안_표식을_제거한다(self):
        # 031 ADR — 검수 마친 골든과 초안이 구분되지 않으면 자동채택 금지 규율이 무의미해진다.
        decided = [{"a": "x", "b": "y", "kind": "references",
                    "_review": "True", "note": "edge"}]
        g = build_golden(decided, [])
        self.assertEqual(set(g["pairs"][0]), {"a", "b", "kind"})
        self.assertNotIn("_review", g["pairs"][0])
        self.assertNotIn("note", g["pairs"][0])

    def test_골든_형식을_지킨다(self):
        g = build_golden([{"a": "x", "b": "y", "kind": "same_domain"}], ["z"])
        self.assertEqual(g["version"], 1)
        self.assertEqual(g["key_type"], "fs_path")
        self.assertEqual(g["isolated"], ["z"])

    def test_parse_golden_이_받아들인다(self):
        # 산출물이 실제 파서를 통과해야 measure 러너가 읽을 수 있다.
        from src.relations.quality.golden import parse_golden
        g = build_golden([{"a": "x", "b": "y", "kind": "same_series"}], ["z"])
        parsed = parse_golden(g)
        self.assertEqual(len(parsed.pairs), 1)
        self.assertEqual(parsed.pairs[0].kind, "same_series")

    def test_빈_결정도_유효한_골든이다(self):
        from src.relations.quality.golden import parse_golden
        parse_golden(build_golden([], []))   # 예외 없이 통과하면 OK


if __name__ == "__main__":
    unittest.main()
