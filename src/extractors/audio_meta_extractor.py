"""오디오 파일 속성 메타(길이·샘플레이트·채널) 추출 — ``src/skills/audio_skill.py`` 가 호출.

내용(전사)이 아닌 컨테이너 속성만 본다. STT 전사·요약은 skill 쪽에서 별도로 처리한다.
"""

from typing import TypedDict

import soundfile as sf

# import librosa

class AudioMeta(TypedDict):
    duration: int
    sample_rate: int
    channels: str

def extract_audio_meta(file_path: str) -> AudioMeta:
    # always_2d=True: 모노/스테레오 무관하게 항상 [samples, channels] 2D로 받아
    # 채널 수를 일관되게 언패킹한다(모노여도 [N,1] 이 되어 분기 불필요).
    y, sr = sf.read(file_path, always_2d=True)   # shape: [samples, channels]
    samples, channels = y.shape
    duration = float(samples / sr)

    return {
        "duration": round(duration, 3),
        "sample_rate": int(sr),
        "channels": int(channels),
    }
