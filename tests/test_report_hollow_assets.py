"""껍데기 자산 감시 리포트의 순수 집계 — DB 없이 판정 로직만 덮는다."""
from __future__ import annotations

import unittest

from scripts.report_hollow_assets import (
    build_report,
    format_report_lines,
    hollow_by_modality,
    hollow_rows,
)


def _row(aid, modality, n, name="x.mp4"):
    return {"asset_id": aid, "fs_path": f"/data/{name}", "modality": modality,
            "embedding_count": n}


class TestHollowRows(unittest.TestCase):
    def test_임베딩_0인_자산만_고른다(self):
        rows = [_row("a1", "video", 0), _row("a2", "video", 3), _row("a3", "text", 0)]
        self.assertEqual([r["asset_id"] for r in hollow_rows(rows)], ["a1", "a3"])

    def test_None_은_0으로_본다(self):
        # LEFT JOIN 결과가 NULL 로 오는 경우 — coalesce 를 믿지 말고 파이썬에서도 방어한다.
        self.assertEqual(len(hollow_rows([_row("a1", "video", None)])), 1)

    def test_unknown_모달리티는_면제다(self):
        # 빈 STT 로 격리된 자산이라 hollow 로 세면 안 된다(그대로 두기로 결정된 것).
        self.assertEqual(hollow_rows([_row("a1", "unknown", 0)]), [])

    def test_빈_입력(self):
        self.assertEqual(hollow_rows([]), [])


class TestHollowByModality(unittest.TestCase):
    def test_모달리티별로_hollow와_전체를_함께_센다(self):
        # 분모가 없으면 "3건"이 심각한지 판단할 수 없다.
        rows = [_row("a1", "video", 0), _row("a2", "video", 5),
                _row("a3", "text", 2), _row("a4", "text", 0), _row("a5", "text", 1)]
        self.assertEqual(hollow_by_modality(rows), {"video": (1, 2), "text": (1, 3)})

    def test_면제_모달리티는_키에_없다(self):
        got = hollow_by_modality([_row("a1", "unknown", 0), _row("a2", "video", 1)])
        self.assertNotIn("unknown", got)
        self.assertEqual(got, {"video": (0, 1)})

    def test_모달리티는_정렬돼_나온다(self):
        rows = [_row("a1", "video", 1), _row("a2", "audio", 1), _row("a3", "image", 1)]
        self.assertEqual(list(hollow_by_modality(rows)), ["audio", "image", "video"])


class TestBuildReport(unittest.TestCase):
    def test_hollow_0건이면_healthy(self):
        r = build_report([_row("a1", "video", 3), _row("a2", "text", 1)])
        self.assertTrue(r["healthy"])
        self.assertEqual(r["hollow_total"], 0)

    def test_한_건이라도_있으면_healthy_아니다(self):
        # 비율 임계를 두지 않는다 — 1건도 064 재개 신호다.
        r = build_report([_row("a1", "video", 0)] + [_row(f"b{i}", "text", 1) for i in range(999)])
        self.assertFalse(r["healthy"])
        self.assertEqual(r["hollow_total"], 1)

    def test_사례는_파일명만_남긴다(self):
        # 전체 경로를 로그에 남길 필요가 없다.
        r = build_report([_row("a1", "video", 0, name="비밀폴더침해.mp4")])
        self.assertEqual(r["samples"][0]["name"], "비밀폴더침해.mp4")
        self.assertNotIn("/data/", r["samples"][0]["name"])

    def test_사례_수는_상한을_지킨다(self):
        rows = [_row(f"a{i}", "video", 0) for i in range(50)]
        self.assertEqual(len(build_report(rows, sample_limit=3)["samples"]), 3)

    def test_면제_모달리티는_분모에서도_빠진다(self):
        r = build_report([_row("a1", "unknown", 0), _row("a2", "video", 1)])
        self.assertEqual(r["asset_total"], 1)

    def test_빈_입력은_healthy(self):
        r = build_report([])
        self.assertTrue(r["healthy"])
        self.assertEqual(r["asset_total"], 0)


class TestFormatReport(unittest.TestCase):
    def test_정상이면_064_재개_문구가_없다(self):
        lines = format_report_lines(build_report([_row("a1", "video", 1)]))
        self.assertTrue(any("hollow 0건" in x for x in lines))
        self.assertFalse(any("064" in x for x in lines))

    def test_hollow_있으면_064_재개_신호를_안내한다(self):
        lines = format_report_lines(build_report([_row("a1", "video", 0)]))
        self.assertTrue(any("064" in x for x in lines))

    def test_분모가_0이어도_나눗셈에서_터지지_않는다(self):
        format_report_lines(build_report([]))   # ZeroDivisionError 없이 통과하면 OK


if __name__ == "__main__":
    unittest.main()
