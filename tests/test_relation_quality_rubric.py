"""루브릭 상수 — 문구와 버전이 함께 움직이는지 봉인한다."""
import hashlib
import unittest

from src.relations.quality.rubric import (
    RUBRIC_KO_V1,
    RUBRIC_KO_V1_SHA256,
    RUBRIC_VERSION,
)


class TestRubric(unittest.TestCase):
    def test_세_판정값이_모두_정의돼_있다(self):
        for label in ("strong", "weak", "none"):
            self.assertIn(f'"{label}"', RUBRIC_KO_V1)

    def test_버전은_비어있지_않다(self):
        self.assertTrue(RUBRIC_VERSION.strip())

    def test_문구를_고치면_해시와_버전을_함께_갱신해야_한다(self):
        # 루브릭이 바뀌었는데 버전이 그대로면 과거 판정과 비교가 불가능해진다 —
        # 그 비교 가능성이 이 모듈의 존재 이유라서 해시로 잠근다.
        actual = hashlib.sha256(RUBRIC_KO_V1.encode("utf-8")).hexdigest()
        self.assertEqual(
            actual, RUBRIC_KO_V1_SHA256,
            "루브릭 문구가 바뀌었다. RUBRIC_KO_V1_SHA256 과 RUBRIC_VERSION 을 함께 갱신하라.")


if __name__ == "__main__":
    unittest.main()
