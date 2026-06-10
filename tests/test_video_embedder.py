"""video_embedder 단위 테스트 — 키프레임 CLIP 임베딩 결과 형태. DB·모델 불필요 경로만."""

from __future__ import annotations

import unittest

from src.embedders.video_embedder import embed_video_keyframes_clip


class TestEmbedVideoKeyframesClip(unittest.TestCase):
    def test_empty_frames_returns_empty_keyframes(self) -> None:
        # 빈 입력은 모델 로드 없이 빈 keyframes 만 반환한다.
        # 소비자 없는 영상 단일 집계 벡터(clip_video_embedding)는 결과에 존재하지 않는다(데드코드 정리 고정).
        out = embed_video_keyframes_clip([])
        self.assertEqual(out, {"keyframes": []})
        self.assertNotIn("clip_video_embedding", out)


if __name__ == "__main__":
    unittest.main()
