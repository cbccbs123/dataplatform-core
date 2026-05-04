from typing import TypedDict
import numpy as np
import soundfile as sf
# import librosa

class AudioMeta(TypedDict):
    duration: int
    sample_rate: int
    channels: str

def extract_audio_meta(file_path: str) -> AudioMeta:
    y, sr = sf.read(file_path, always_2d=True)   # shape: [samples, channels]
    samples, channels = y.shape
    duration = float(samples / sr)

    return {
        "duration": round(duration, 3),
        "sample_rate": int(sr),
        "channels": int(channels),
    }