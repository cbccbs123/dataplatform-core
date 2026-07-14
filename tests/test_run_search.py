"""006 검색 CLI 진입점(run_search) — 인자 파싱·매핑 단위 테스트(네트워크/DB 없음).

CLI 의 순수 부분(모달리티 파싱, 인자→search_hybrid 매핑)만 검증한다. 실제 검색은
``search_fn`` 주입으로 대체한다(실 동작은 T016 e2e).
"""

from __future__ import annotations

import argparse
import unittest

from src.app import run_search


class TestParseModalities(unittest.TestCase):
    def test_none_and_blank_return_none(self) -> None:
        self.assertIsNone(run_search._parse_modalities(None))
        self.assertIsNone(run_search._parse_modalities(""))
        self.assertIsNone(run_search._parse_modalities("   "))

    def test_comma_split_and_strip(self) -> None:
        self.assertEqual(run_search._parse_modalities("text,image"), ["text", "image"])
        self.assertEqual(run_search._parse_modalities(" text , , image "), ["text", "image"])


class TestRunMapsArgs(unittest.TestCase):
    def test_run_maps_args_to_search_kwargs(self) -> None:
        captured: dict[str, object] = {}

        def fake_search(query: str, **kw: object) -> dict[str, object]:
            captured["query"] = query
            captured.update(kw)
            return {"query": query, "results": {}, "meta": {}}

        ns = argparse.Namespace(query="질의", modalities="text,image", limit=5)
        out = run_search._run(ns, search_fn=fake_search)

        self.assertEqual(captured["query"], "질의")
        self.assertEqual(captured["modalities"], ["text", "image"])
        self.assertEqual(captured["limit_per_bucket"], 5)
        # 069 US-C: 037 로 no-op 였던 text_hybrid_alpha·min_scores 는 더 이상 전달하지 않는다.
        self.assertNotIn("text_hybrid_alpha", captured)
        self.assertNotIn("min_scores", captured)
        self.assertEqual(out["query"], "질의")

    def test_run_no_modalities_passes_none(self) -> None:
        captured: dict[str, object] = {}

        def fake_search(query: str, **kw: object) -> dict[str, object]:
            captured.update(kw)
            return {}

        ns = argparse.Namespace(query="질의", modalities=None, limit=20)
        run_search._run(ns, search_fn=fake_search)
        self.assertIsNone(captured["modalities"])


if __name__ == "__main__":
    unittest.main()
