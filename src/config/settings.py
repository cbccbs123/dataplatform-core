from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class PipelineSettings:
    profile: Literal["dev", "prod"]
    meta_model: str
    openai_base_url: str
    openai_api_key: str
    summary_max_chars: int
    top_k_keywords: int
    chunk_size: int
    overlap_size: int
    encoding: str
    text_embedding_model: str
    text_embedding_chunk_size: int
    text_embedding_normalize: bool
    video_max_keyframes: int


_SETTINGS: PipelineSettings | None = None


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"필수 환경변수 누락: {name}")
    return value.strip()


def _require_env_int(name: str) -> int:
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"정수 환경변수 형식 오류: {name}={raw!r}") from e


def _require_env_bool(name: str) -> bool:
    raw = _require_env(name).lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"불리언 환경변수 형식 오류: {name}={raw!r}")


def _env_int_default(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError as e:
        raise ValueError(f"정수 환경변수 형식 오류: {name}={raw!r}") from e


def _build_settings(profile: Literal["dev", "prod"]) -> PipelineSettings:
    return PipelineSettings(
        profile=profile,
        meta_model=_require_env("META_MODEL"),
        openai_base_url=_require_env("OPENAI_BASE_URL"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        summary_max_chars=_require_env_int("SUMMARY_MAX_CHARS"),
        top_k_keywords=_require_env_int("TOP_K_KEYWORDS"),
        chunk_size=_require_env_int("CHUNK_SIZE"),
        overlap_size=_require_env_int("OVERLAP_SIZE"),
        encoding=_require_env("ENCODING"),
        text_embedding_model=_require_env("TEXT_EMBED_MODEL"),
        text_embedding_chunk_size=_require_env_int("TEXT_EMBED_CHUNK_SIZE"),
        text_embedding_normalize=_require_env_bool("TEXT_EMBED_NORMALIZE"),
        video_max_keyframes=(
            48 if (vk := _env_int_default("VIDEO_MAX_KEYFRAMES", 48)) <= 0 else vk
        ),
    )


def init_settings(profile: Literal["dev", "prod"]) -> PipelineSettings:
    global _SETTINGS
    _SETTINGS = _build_settings(profile)
    return _SETTINGS


def get_current_settings() -> PipelineSettings:
    if _SETTINGS is None:
        raise RuntimeError("settings가 초기화되지 않았습니다. 먼저 init_settings(profile)를 호출하세요.")
    return _SETTINGS