"""자산 썸네일 생성 단위 테스트 (057-후속) — PIL 로 소형 이미지를 만들어 순수 검증(실 OS·DB 불필요).

검증 의도
    - 이미지 → JPEG 바이트, 최대 변 ``THUMB_MAX_DIM`` 이내(비율 보존).
    - 결정성(헌법 3조): 동일 입력 → 동일 바이트.
    - 비대상 modality(audio/text/unknown)·빈/부재 경로 → ``None``(엔드포인트 404 → 프론트 아이콘).
    (영상 경로는 실 영상 파일이 필요해 여기선 modality 게이트만; 실값은 엔드포인트/실DB 게이트에서.)
"""
from __future__ import annotations

import os
import tempfile
import unittest
from io import BytesIO

from src.portal.thumbnail import THUMB_MAX_DIM, generate_thumbnail


def _make_png(path: str, size: tuple[int, int] = (800, 600)) -> None:
    from PIL import Image

    Image.new("RGB", size, (120, 60, 200)).save(path, "PNG")


class GenerateThumbnailTest(unittest.TestCase):
    def test_image_returns_jpeg_within_max_dim_ratio_preserved(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            _make_png(p, (800, 600))
            data = generate_thumbnail(p, "image")
            self.assertIsNotNone(data)
            im = Image.open(BytesIO(data))
            self.assertEqual(im.format, "JPEG")
            self.assertLessEqual(max(im.size), THUMB_MAX_DIM)
            self.assertEqual(im.size, (THUMB_MAX_DIM, 240))  # 800x600 → 320x240(비율 보존)

    def test_deterministic_same_bytes(self) -> None:
        # 동일 입력 → 동일 바이트(결정성·헌법 3조). 고정 리샘플(LANCZOS)·JPEG 파라미터.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.png")
            _make_png(p, (640, 480))
            self.assertEqual(generate_thumbnail(p, "image"), generate_thumbnail(p, "image"))

    def test_small_image_not_upscaled(self) -> None:
        # 원본이 상한보다 작으면 확대하지 않는다(thumbnail 은 축소 전용).
        from PIL import Image

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.png")
            _make_png(p, (100, 80))
            im = Image.open(BytesIO(generate_thumbnail(p, "image")))
            self.assertEqual(im.size, (100, 80))

    def test_non_thumbnailable_modality_none(self) -> None:
        self.assertIsNone(generate_thumbnail("/x/a.txt", "text"))
        self.assertIsNone(generate_thumbnail("/x/a.mp3", "audio"))
        self.assertIsNone(generate_thumbnail("/x/a.bin", "unknown"))
        self.assertIsNone(generate_thumbnail("/x/a.png", None))

    def test_missing_or_empty_path_none(self) -> None:
        self.assertIsNone(generate_thumbnail("", "image"))
        self.assertIsNone(generate_thumbnail("/nonexistent/nope.png", "image"))  # open 실패 → 격리 None


if __name__ == "__main__":
    unittest.main()
