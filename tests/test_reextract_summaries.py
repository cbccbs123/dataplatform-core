"""026 — 재추출 배치 스크립트의 순수 로직 단위(리뷰 후속).

LLM·DB 없이 검증 가능한 부분만: STT 입력 해소(`_audio_text` 사이드카 폴백)와
모달리티 디스패치의 skip 경계(`_reextract_one` — 입력 부재 → None). 실제 재추출
실행은 G5 배치(사람 게이트)에서 수행·검증했다(ADR 실측).
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reextract_summaries.py"
_spec = importlib.util.spec_from_file_location("reextract_summaries", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class AudioTextTest(unittest.TestCase):
    def test_ext_meta_stt_first(self) -> None:
        # ext_meta.stt 가 있으면 사이드카를 보지 않고 그것을 쓴다(whisper 재실행 금지 경로).
        self.assertEqual(_mod._audio_text({"stt": " 전사 텍스트 "}, "/없는/경로.mp3"), "전사 텍스트")

    def test_sidecar_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "a.mp3"
            Path(f"{audio}.stt.txt").write_text("사이드카 전사", encoding="utf-8")
            self.assertEqual(_mod._audio_text({}, str(audio)), "사이드카 전사")

    def test_empty_when_no_source(self) -> None:
        self.assertEqual(_mod._audio_text({}, "/없는/경로.mp3"), "")


class ReextractDispatchSkipTest(unittest.TestCase):
    """입력 부재 시 None(skip) — LLM 호출 전에 끊어져 순수하게 검증 가능한 경계만."""

    def test_unknown_modality_none(self) -> None:
        self.assertIsNone(_mod._reextract_one("unknown", "/x", {}))

    def test_video_without_keyframes_none(self) -> None:
        # 장면 캡션이 없으면 재합성 불가 → skip(VLM 재실행으로 폴백하지 않음 — 2단계 영역).
        self.assertIsNone(_mod._reextract_one("video", "/x.mp4", {}))
        self.assertIsNone(_mod._reextract_one("video", "/x.mp4", {"keyframes": []}))

    def test_doc_missing_file_none(self) -> None:
        self.assertIsNone(_mod._reextract_one("txt", "/없는/파일.txt", {}))

    def test_image_missing_file_none(self) -> None:
        self.assertIsNone(_mod._reextract_one("image", "/없는/파일.jpg", {}))

    def test_audio_without_text_none(self) -> None:
        self.assertIsNone(_mod._reextract_one("audio", "/없는/경로.mp3", {}))


if __name__ == "__main__":
    unittest.main()
