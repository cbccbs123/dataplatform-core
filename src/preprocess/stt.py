"""
음성 파일 → 텍스트(STT). 온프레미스 로컬 추론만 사용한다(외부 API 미호출).

``faster-whisper``(CTranslate2 백엔드)로 Whisper 모델을 로컬에서 돌린다. 기본은 한국어(``ko``),
``small`` 모델, CPU + ``int8`` 양자화이며, 인자로 GPU(``cuda``/``float16``)·모델 크기를 바꿀 수 있다.
``vad_filter=True`` 로 무음 구간을 걸러 환각을 줄이고, 모델이 낸 세그먼트 텍스트를 공백으로 이어
'전체 텍스트'만 돌려준다(타임스탬프·세그먼트 경계는 보존하지 않음).

audio_skill 이 이 전체 텍스트를 받아 요약·임베딩(media_chunks 의 STT 텍스트 청크)으로 잇는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from faster_whisper import WhisperModel


class TranscriptionResult(TypedDict):
    """전사 결과 — 세그먼트를 이어 붙인 전체 텍스트만 담는다."""

    text: str


def transcribe_audio_local(
    file_path: str | Path,
    *,
    model_size: str = "small",   # Whisper-small
    language: str | None = "ko",
    device: str = "cpu",         # GPU면 "cuda"
    compute_type: str = "int8",  # GPU면 "float16" 권장
) -> TranscriptionResult:
    """로컬 Whisper 로 ``file_path`` 를 전사해 전체 텍스트(``{"text": ...}``)를 반환한다."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments_iter, _info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,
        beam_size=5,
    )
    texts: list[str] = []
    for seg in segments_iter:
        t = (seg.text or "").strip()
        texts.append(t)
    return {"text": " ".join(texts).strip()}
