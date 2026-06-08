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
    # 017 A/B: BGE-M3 채널('st_bge')용 임베딩 모델. _require_env 가 아닌 선택 필드 —
    # 미설정 시 기본 'BAAI/bge-m3'(기존 text_embedding_model=KoSimCSE 와 별개, 동작 무변경).
    text_embedding_model_bge: str
    # 018: 운영 텍스트 임베딩 활성 채널. 적재·검색·관계가 'st' 하드코딩 대신 이 단일 출처를 참조해
    # 모델을 토글한다. _env_str_default 선택 필드 — 미설정 시 기본 'st'(KoSimCSE) → 동작 무변경.
    active_embed_channel: str
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
    # 경로 신호(path_signal) 후보의 별도 한도(008, C-2). 임베딩 top_k 와 독립적으로 LIMIT 해
    # 동일 폴더 폭주를 차단한다. union 총 후보 ≤ relation_top_k + relation_path_top_k.
    relation_path_top_k: int
    # 관계 재시도/미해소 큐(relation_resolution)의 재시도 상한(009). attempts 가 이 값에 도달하면
    # decide_resolution_status 가 failed(DLQ)로 승격한다. run_relations --retry 가 소비 (기본 3).
    relation_retry_max_attempts: int
    # 검색 결과 적합도 하한(모달리티별). 0.0=비활성. relations 의 relation_min_sim 과 같은 성격의 게이트.
    search_min_scores: dict[str, float]


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


def _env_str_default(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_float_default(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError as e:
        raise ValueError(f"실수 환경변수 형식 오류: {name}={raw!r}") from e


# 검색이 쓰는 모달리티 버킷. relations 와 달리 모달리티마다 점수 스케일이 달라(텍스트 하이브리드 vs
# 시각 2단계) 임계값을 모달리티별로 둔다.
_SEARCH_MIN_SCORE_MODALITIES = ("text", "image", "video", "audio")


def resolve_search_min_scores() -> dict[str, float]:
    """모달리티 → 검색 적합도 하한 임계값. 공통 ``SEARCH_MIN_SCORE``(기본 0.0)를 각 모달리티
    기본값으로 쓰고, ``SEARCH_MIN_SCORE_<MODALITY>`` 가 있으면 덮는 2단 폴백. 0.0 이면 비활성."""
    common = _env_float_default("SEARCH_MIN_SCORE", 0.0)
    return {
        m: _env_float_default(f"SEARCH_MIN_SCORE_{m.upper()}", common)
        for m in _SEARCH_MIN_SCORE_MODALITIES
    }


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
        text_embedding_model_bge=_env_str_default("TEXT_EMBED_MODEL_BGE", "BAAI/bge-m3"),
        active_embed_channel=_env_str_default("EMBED_ACTIVE_CHANNEL", "st"),
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
        relation_path_top_k=_env_int_default("RELATION_PATH_TOP_K", 10),
        relation_retry_max_attempts=_env_int_default("RELATION_RETRY_MAX_ATTEMPTS", 3),
        search_min_scores=resolve_search_min_scores(),
    )


def init_settings(profile: Literal["dev", "prod"]) -> PipelineSettings:
    global _SETTINGS
    _SETTINGS = _build_settings(profile)
    return _SETTINGS


def get_current_settings() -> PipelineSettings:
    if _SETTINGS is None:
        raise RuntimeError("settings가 초기화되지 않았습니다. 먼저 init_settings(profile)를 호출하세요.")
    return _SETTINGS


def model_for_channel(channel: str, settings: PipelineSettings | None = None) -> str:
    """텍스트 임베딩 채널 → 질의 임베딩 모델 매핑(017 A/B). 질의-문서 모델을 일치(FR-004)시키는 단일 출처.

    ``'st'``=KoSimCSE(기존), ``'st_bge'``=BGE-M3. ``settings`` 미지정 시 활성 설정을 사용한다
    (테스트는 ``settings`` 를 주입해 순수 단위로 검증). 미지원 채널은 잘못된 모델로 검색하지 않도록
    즉시 ``ValueError`` 로 차단한다(시각 'clip' 채널은 본 텍스트 매핑 대상이 아님)."""
    cfg = settings if settings is not None else get_current_settings()
    mapping = {
        "st": cfg.text_embedding_model,
        "st_bge": cfg.text_embedding_model_bge,
    }
    try:
        return mapping[channel]
    except KeyError:
        raise ValueError(
            f"지원하지 않는 텍스트 임베딩 채널: {channel!r} (지원: {sorted(mapping)})"
        ) from None


def active_embed_channel(settings: PipelineSettings | None = None) -> str:
    """운영 텍스트 임베딩 활성 채널(018). 적재·검색·관계가 공유하는 단일 출처.

    ``settings`` 미지정 시 활성 설정을 사용한다(테스트는 ``settings`` 주입으로 순수 단위 검증).
    기본값은 ``'st'``(KoSimCSE) — 회귀 0."""
    cfg = settings if settings is not None else get_current_settings()
    return cfg.active_embed_channel


def active_embed_model(settings: PipelineSettings | None = None) -> str:
    """활성 채널의 임베딩 모델(018). ``active_embed_channel`` → ``model_for_channel`` 합성.

    기본 active='st' → KoSimCSE(``text_embedding_model``), 'st_bge' → BGE-M3. 미지원 활성 채널은
    ``model_for_channel`` 이 즉시 ``ValueError`` 로 차단한다(잘못된 모델 사용 방지)."""
    return model_for_channel(active_embed_channel(settings), settings)
