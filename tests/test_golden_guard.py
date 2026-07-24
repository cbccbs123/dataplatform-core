"""025 G3 — 코퍼스-골든 정합 가드(순수) 단위.

신규 토픽 자산이 코퍼스에 추가되면 골든 질의도 추가되어야 한다(운영 규칙)는 것을
순수 함수 `uncovered_topics` 로 강제한다 — 미커버 토픽이 생기면 gated e2e 가 실패한다.
"""

from __future__ import annotations

import unittest

from src.search.golden_guard import topic_of_filename, uncovered_topics


class TopicOfFilenameTest(unittest.TestCase):
    def test_first_token_of_slug(self) -> None:
        # 구형 <주제>_<11자ID>_<제목> → 첫 토큰이 주제.
        self.assertEqual(topic_of_filename("등산_입문_TUWlGnSstVI_제목.mp4"), "등산")
        self.assertEqual(topic_of_filename("무선_충전기_x.jpg"), "무선")

    def test_source_prefix_uses_second_token(self) -> None:
        # 신규 youtube_/wikipedia_ 출처-prefix → 2번째 토큰이 진짜 주제(prefix 는 주제 아님).
        self.assertEqual(topic_of_filename("youtube_사막_3bTA2c2n2QI.jpg"), "사막")
        self.assertEqual(topic_of_filename("wikipedia_고려청자_1031019.txt"), "고려청자")
        # 주제에 공백이 있어도 한 필드(밑줄로만 분리).
        self.assertEqual(topic_of_filename("youtube_기후 변화_3CHPt7zk5fE.mp4"), "기후 변화")

    def test_trailing_paren_topic(self) -> None:
        # 재수집 명명 <uuid>__<제목>_(주제).ext → 끝 괄호가 주제(토큰 위치 불안정 → 표식 우선).
        self.assertEqual(
            topic_of_filename("018f0000-0000-7000-8000-000000000273__Yoke_and_Arrows_(전통주).svg"),
            "전통주",
        )
        self.assertEqual(
            topic_of_filename("018f0000-0000-7000-8000-000000000276__서귀포 열대우림 🌴_(열대우림).mp4"),
            "열대우림",
        )
        # 괄호 없는 신규 youtube(+uuid 접두)는 접두 제거 후 2번째 토큰.
        self.assertEqual(
            topic_of_filename("018f0000-0000-7000-8000-000000000274__youtube_빙하_6uWBi3GrRYM.mp3"),
            "빙하",
        )

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
