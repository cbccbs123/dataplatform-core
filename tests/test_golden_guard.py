"""025 G3 — 코퍼스-골든 정합 가드(순수) 단위.

신규 토픽 자산이 코퍼스에 추가되면 골든 질의도 추가되어야 한다(운영 규칙)는 것을
순수 함수 `uncovered_topics` 로 강제한다 — 미커버 토픽이 생기면 gated e2e 가 실패한다.
"""

from __future__ import annotations

import unittest

from src.search.golden_guard import topic_of_filename, uncovered_topics


class TopicOfFilenameTest(unittest.TestCase):
    def test_first_token_of_slug(self) -> None:
        self.assertEqual(topic_of_filename("등산_입문_TUWlGnSstVI_제목.mp4"), "등산")
        self.assertEqual(topic_of_filename("무선_충전기_x.jpg"), "무선")

    def test_no_underscore_returns_stem(self) -> None:
        self.assertEqual(topic_of_filename("manifest.json"), "manifest")

    def test_empty_safe(self) -> None:
        self.assertEqual(topic_of_filename(""), "")


class UncoveredTopicsTest(unittest.TestCase):
    def test_full_coverage_returns_empty(self) -> None:
        self.assertEqual(uncovered_topics({"등산", "주식"}, {"등산", "주식", "수영"}), [])

    def test_new_topic_detected(self) -> None:
        # 코퍼스에 '겨울낚시' 토픽 자산이 새로 들어왔는데 골든 질의가 없다 → 검출(정렬·결정적).
        self.assertEqual(
            uncovered_topics({"등산", "겨울낚시", "주식"}, {"등산", "주식"}), ["겨울낚시"]
        )

    def test_deterministic_sorted(self) -> None:
        self.assertEqual(uncovered_topics({"c", "a", "b"}, set()), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
