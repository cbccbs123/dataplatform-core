"""
음성 파일 → 텍스트(STT).

OpenAI 호환 ``/v1/audio/transcriptions`` 엔드포인트를 사용한다.
(``get_current_settings()``의 ``openai_base_url`` / ``openai_api_key``)

대표 구간(“어디가 핵심 발화인가?”)은 보통 아래처럼 나눈다.

1. **Whisper 세그먼트(이 모듈)** — ``verbose_json`` + ``timestamp_granularities`` 로
   구간별 텍스트·시간이 오면, 가장 긴 구간·텍스트가 많은 구간 등으로 고른다.
2. **VAD** — WebRTC VAD, silero-vad, pyannote 등으로 음성 구간만 잘라낸 뒤
   그 구간만 STT하거나, 가장 긴 음성 구간을 “대표”로 쓴다.
3. **에너지/RMS** — 짧은 윈도우 RMS가 높은 구간을 후보로 쓰는 휴리스틱(가벼움, 부정확할 수 있음).
4. **ffmpeg silencedetect** — 무음 구간 경계로 쪼갠 뒤 가장 긴 덩어리 선택.

게이트웨이가 세그먼트 타임스탬프를 지원하지 않으면 ``segments`` 는 빈 리스트이고
``text`` 만 채워진다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from faster_whisper import WhisperModel


class TranscriptionResult(TypedDict):
    """전체 텍스트와 구간 목록(가능할 때만 채움)."""

    text: str


def transcribe_audio_local(
    file_path: str | Path,
    *,
    model_size: str = "small",   # Whisper-small
    language: str | None = "ko",
    device: str = "cpu",         # GPU면 "cuda"
    compute_type: str = "int8",  # GPU면 "float16" 권장
) -> TranscriptionResult:
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
