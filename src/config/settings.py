"""``.env.*`` 파이프라인 설정(임베딩, CLIP 라벨 상한 등). ``init_settings`` 후 ``get_current_settings`` 로 조회."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class PipelineSettings:
    """파이프라인 실행 설정. ``init_settings(profile)`` 이 한 번만 생성하며 이후 변경 불가(frozen).

    ``_require_env*`` 로 읽는 필드는 누락 시 즉시 ValueError — 필수 환경변수.
    ``_env_*_default`` 로 읽는 필드는 미설정 시 하드코드 기본값 사용 — 선택 환경변수.
    """

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
    image_labels_search_top_k: int
    video_keyframe_labels_search_top_k: int
    image_labels_meta_top_k: int
    video_keyframe_labels_meta_top_k: int
    labels_score_min: float
    relation_top_k: int
    # 아래 두 필드는 이번 브랜치(relations-catalog-slim)에서 추가된 관계 제안 품질 게이트.
    relation_min_sim: float       # 후보 코사인 유사도 하한 — 이 미만은 LLM 에 넣지 않음 (기본 0.2)
    relation_auto_approve_min: float  # 이 이상이면 HITL 없이 자동 승인 (기본 0.9)


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


def _env_float_default(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError as e:
        raise ValueError(f"실수 환경변수 형식 오류: {name}={raw!r}") from e


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
        image_labels_search_top_k=_env_int_default("IMAGE_LABELS_SEARCH_TOP_K", 5),
        video_keyframe_labels_search_top_k=_env_int_default(
            "VIDEO_KEYFRAME_LABELS_SEARCH_TOP_K", 3
        ),
        image_labels_meta_top_k=_env_int_default("IMAGE_LABELS_META_TOP_K", 10),
        video_keyframe_labels_meta_top_k=_env_int_default(
            "VIDEO_KEYFRAME_LABELS_META_TOP_K", 5
        ),
        labels_score_min=_env_float_default("LABELS_SCORE_MIN", 0.1),
        relation_top_k=_env_int_default("RELATION_TOP_K", 10),
        relation_min_sim=_env_float_default("RELATION_MIN_SIM", 0.2),
        relation_auto_approve_min=_env_float_default("RELATION_AUTO_APPROVE_MIN", 0.9),
    )


def init_settings(profile: Literal["dev", "prod"]) -> PipelineSettings:
    global _SETTINGS
    _SETTINGS = _build_settings(profile)
    return _SETTINGS


def get_current_settings() -> PipelineSettings:
    if _SETTINGS is None:
        raise RuntimeError("settings가 초기화되지 않았습니다. 먼저 init_settings(profile)를 호출하세요.")
    return _SETTINGS