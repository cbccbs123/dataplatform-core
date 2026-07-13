"""073 — about_or_filter 순수 단위(usable-noun 상대 DF·amatch/kmatch·fail-safe·순서 보존)."""

from __future__ import annotations

import unittest
from typing import Any

from src.search.about_filter import about_or_filter


def _row(rid: str, about: list[str] | None = None, kwtext: str | None = None) -> dict[str, Any]:
    r: dict[str, Any] = {"id": rid, "similarity": 0.5}
    if about is not None:
        r["_about"] = about
    if kwtext is not None:
        r["_kwtext"] = kwtext
    return r


class TestAboutOrFilter(unittest.TestCase):
    def test_same_category_confusion_dropped(self) -> None:
        # 표적 케이스(측정 실사례): "기타 연주" — 바이올린(무증거)은 드롭, 기타는 유지.
        rows = [
            _row("guitar", about=["기타"], kwtext="기타 연주법 guitar.txt"),
            _row("violin", about=["바이올린"], kwtext="바이올린 연주 현악기 violin.txt"),
        ]
        out = about_or_filter(rows, "기타 연주")
        self.assertEqual([r["id"] for r in out], ["guitar"])

    def test_common_noun_not_usable_for_kmatch(self) -> None:
        # '연주'는 두 행 모두의 kwtext 에 등장(2/2 > 0.5) → 판별력 없음 → kmatch 에서 제외.
        # 바이올린 행이 '연주' 만으로 살아남으면 안 된다(위 테스트와 같은 데이터로 명시 검증).
        rows = [
            _row("guitar", about=[], kwtext="기타 연주법"),
            _row("violin", about=[], kwtext="바이올린 연주"),
        ]
        out = about_or_filter(rows, "기타 연주")
        self.assertEqual([r["id"] for r in out], ["guitar"])

    def test_amatch_bidirectional_substring(self) -> None:
        # 부분일치 양방향(len≥2): 질의 '고래' ↔ about '고래하목' 매칭(측정 규칙 그대로).
        rows = [
            _row("whale", about=["고래하목"], kwtext="해양 포유류"),
            _row("shark", about=["상어"], kwtext="연골어류"),
        ]
        out = about_or_filter(rows, "고래 울음소리")
        self.assertEqual([r["id"] for r in out], ["whale"])

    def test_single_char_noun_exact_only(self) -> None:
        # 1자 명사('배')는 완전일치만 — '배드민턴'에 부분일치 오매칭 금지.
        rows = [
            _row("ship", about=["배"], kwtext="선박"),
            _row("badminton", about=["배드민턴"], kwtext="라켓 스포츠"),
        ]
        out = about_or_filter(rows, "배 사진")
        self.assertEqual([r["id"] for r in out], ["ship"])

    def test_single_char_noun_excluded_from_kmatch(self) -> None:
        # 리뷰 지적 회귀: 1자 명사는 kmatch(부분일치)에서 제외 — '배'가 '택배'의 글자에 우발 매칭돼
        # 무관 행(택배 안내)이 살아남으면 안 된다. amatch 완전일치('배'==about '배')로만 유지.
        rows = [
            _row("ship", about=["배"], kwtext="선박 항해"),
            _row("courier", about=["택배기사"], kwtext="택배 배송 안내 courier.jpg"),
        ]
        out = about_or_filter(rows, "배 사진")
        self.assertEqual([r["id"] for r in out], ["ship"])

    def test_failsafe_keeps_all_when_nothing_matches(self) -> None:
        # fail-safe(FR-004): 전멸이면 원 행 그대로 — 패러프레이즈 질의(어휘 무겹침) 보호.
        rows = [_row("a", about=["별"], kwtext="천체 사진"), _row("b", about=["망원경"], kwtext="관측 장비")]
        out = about_or_filter(rows, "우주 신비")
        self.assertEqual([r["id"] for r in out], ["a", "b"])

    def test_backcompat_passthrough_without_evidence_keys(self) -> None:
        # 구 색인·mock 행(_about/_kwtext 없음) → 필터 근거 없음 → passthrough(하위호환).
        rows = [{"id": "x", "similarity": 0.9}, {"id": "y", "similarity": 0.1}]
        self.assertEqual(about_or_filter(rows, "기타 연주"), rows)

    def test_empty_inputs_passthrough(self) -> None:
        self.assertEqual(about_or_filter([], "기타"), [])
        rows = [_row("a", about=["기타"], kwtext="")]
        self.assertEqual(about_or_filter(rows, ""), rows)

    def test_order_preserved_drop_only(self) -> None:
        # 드롭만 — 유지 행의 상대 순서·내용 불변(재정렬·점수 변경 없음).
        rows = [
            _row("r1", about=["김치"], kwtext="김치 레시피"),
            _row("r2", about=["된장"], kwtext="된장찌개"),
            _row("r3", about=["김장"], kwtext="김장 배추"),
        ]
        out = about_or_filter(rows, "김치 담그기")
        # r1=amatch(김치)·r3=amatch(김치⊂김장? 아니오 — '김치' in '김장' False; kmatch '김치' in '김장 배추' False)
        # → r3 는 증거 없음. 유지 행 순서는 원래 순서 그대로.
        self.assertEqual([r["id"] for r in out], ["r1"])


if __name__ == "__main__":
    unittest.main()
