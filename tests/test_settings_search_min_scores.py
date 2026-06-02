"""검색 적합도 하한(모달리티별 임계값) 환경변수 해석 — 순수 단위 테스트(필수 env 불필요).

``resolve_search_min_scores`` 는 공통 ``SEARCH_MIN_SCORE`` 를 각 모달리티 기본값으로 쓰고
``SEARCH_MIN_SCORE_<MODALITY>`` 가 있으면 그것으로 덮는 2단 폴백을 수행한다.
``init_settings`` 가 요구하는 17개 필수 env 와 무관하게 이 함수만 독립 검증한다.
"""

from __future__ import annotations

import contextlib
import os
import unittest

from src.config.settings import resolve_search_min_scores

_KEYS = (
    "SEARCH_MIN_SCORE",
    "SEARCH_MIN_SCORE_TEXT",
    "SEARCH_MIN_SCORE_IMAGE",
    "SEARCH_MIN_SCORE_VIDEO",
    "SEARCH_MIN_SCORE_AUDIO",
)


@contextlib.contextmanager
def _env(**overrides: str):
    """관련 SEARCH_MIN_SCORE* 키만 깨끗이 비운 뒤 ``overrides`` 만 설정하고 복원한다."""
    saved = {k: os.environ.pop(k, None) for k in _KEYS}
    try:
        os.environ.update({k: str(v) for k, v in overrides.items()})
        yield
    finally:
        for k in _KEYS:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


class TestResolveSearchMinScores(unittest.TestCase):
    def test_default_is_zero_for_all_modalities(self) -> None:
        with _env():
            scores = resolve_search_min_scores()
        self.assertEqual(scores, {"text": 0.0, "image": 0.0, "video": 0.0, "audio": 0.0})

    def test_common_applies_to_all_modalities(self) -> None:
        with _env(SEARCH_MIN_SCORE="0.2"):
            scores = resolve_search_min_scores()
        self.assertEqual(scores, {"text": 0.2, "image": 0.2, "video": 0.2, "audio": 0.2})

    def test_per_modality_override_beats_common(self) -> None:
        with _env(SEARCH_MIN_SCORE="0.2", SEARCH_MIN_SCORE_IMAGE="0.05"):
            scores = resolve_search_min_scores()
        self.assertEqual(scores["image"], 0.05)
        self.assertEqual(scores["text"], 0.2)
        self.assertEqual(scores["video"], 0.2)
        self.assertEqual(scores["audio"], 0.2)

    def test_per_modality_without_common_falls_back_to_zero(self) -> None:
        with _env(SEARCH_MIN_SCORE_TEXT="0.3"):
            scores = resolve_search_min_scores()
        self.assertEqual(scores["text"], 0.3)
        self.assertEqual(scores["image"], 0.0)


if __name__ == "__main__":
    unittest.main()
