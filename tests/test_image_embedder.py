"""069 US-B B1(P2-1) — CLIP 한글 라벨 후보/메타 정렬 결정화(헌법 3조).

실 CLIP·네트워크 0. 두 순수 함수만 검증:
  - ``normalize_korean_label_candidates``: 입력 순서 보존 dedup(``dict.fromkeys``) — 동일 입력 동일 출력.
  - ``clip_zero_shot_ko_meta_items``: 동점 score 는 label 문자열 2차키로 결정적 정렬(top-k 컷 결정화).
"""

from __future__ import annotations

import unittest

from src.embedders.image_embedder import (
    clip_zero_shot_ko_meta_items,
    normalize_korean_label_candidates,
)


class TestNormalizeKoreanLabelCandidates(unittest.TestCase):
    def test_dedup_preserves_input_order(self) -> None:
        # dict.fromkeys → 첫 등장 순서 유지(set 은 해시 순서라 비결정적). trim 도 함께 검증.
        out = normalize_korean_label_candidates(["개", " 고양이 ", "개", "토끼", "고양이"])
        self.assertEqual(out, ["개", "고양이", "토끼"])

    def test_blank_dropped(self) -> None:
        self.assertEqual(normalize_korean_label_candidates(["", "  ", "새"]), ["새"])

    def test_same_input_same_output(self) -> None:
        # 헌법 3조: 동일 입력 반복 실행 시 동일 출력(순서 포함).
        labels = ["다람쥐", "여우", "다람쥐", "곰", "여우", "사슴"]
        self.assertEqual(
            normalize_korean_label_candidates(labels),
            normalize_korean_label_candidates(labels),
        )
        self.assertEqual(normalize_korean_label_candidates(labels), ["다람쥐", "여우", "곰", "사슴"])


class TestClipZeroShotKoMetaItemsTiebreak(unittest.TestCase):
    def test_ties_broken_by_label_string(self) -> None:
        # 동점(0.5) 두 라벨은 label 문자열 오름차순으로 고정 — '가방' < '나비'(유니코드).
        # 삽입 순서가 나비→가방 이라 2차키 없으면 안정정렬로 [나비, 가방] 이 되는 것을 막는다.
        items = clip_zero_shot_ko_meta_items({"나비": 0.5, "가방": 0.5, "다리": 0.3})
        labels = [it["label"] for it in items]
        self.assertEqual(labels, ["가방", "나비", "다리"])

    def test_score_desc_primary(self) -> None:
        items = clip_zero_shot_ko_meta_items({"저": 0.1, "고": 0.9, "중": 0.5})
        self.assertEqual([it["label"] for it in items], ["고", "중", "저"])


if __name__ == "__main__":
    unittest.main()
