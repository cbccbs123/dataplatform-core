"""``.env.*`` 파이프라인 설정(임베딩, CLIP 라벨 상한 등). ``init_settings`` 후 ``get_current_settings`` 로 조회."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

_LOG = logging.getLogger(__name__)


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
    # 019: per-asset 청크 집계 방식(검색 시점 산식). 검색 경로(ST 하이브리드·시각 2단계)가 흩어진
    # MAX 하드코딩 대신 이 단일 출처(chunk_agg_config)를 참조한다. 선택 필드 —
    # 미설정 시 기본 'max'(=기존 MAX 집계와 동치) → 회귀 0(SC-001). 'topk_mean'/'mix' 는 측정용 토글.
    chunk_agg: str
    chunk_agg_k: int        # topk_mean 상위 k(기본 3)
    chunk_agg_mix_w: float  # mix 가중치 w: w*MAX + (1-w)*AVG (기본 0.5)
    # 020: OpenSearch 동기화(검색 엔진 도입·CQRS). 모두 선택 필드(018/019 동형) — 미설정 시 기존
    # 동작 무영향(SC-001). opensearch_sync_enabled 기본 False 면 run_ingest 증분 훅이 실행되지 않고
    # opensearch_sync/opensearch-py 를 import 조차 하지 않는다. url/index 는 증분 훅·복구 도구가 참조.
    opensearch_url: str
    opensearch_index: str
    opensearch_sync_enabled: bool
    # 021: 검색 read path 백엔드 전환(CQRS·검색 엔진 도입). 모두 선택 필드(020 동형) — 미설정 시 기존
    # 동작 무영향(SC-001). search_backend 기본 'pg' 면 search_service.search_hybrid 가 현 media_search
    # 경로 그대로(회귀 0). 화이트리스트 밖 값은 _build_settings(=init_settings)에서 즉시 ValueError로
    # 차단한다(fail-fast — 런타임까지 오설정이 숨지 않게, 백로그 '설정 fail-late' 교정).
    # opensearch_search_pipeline/opensearch_fusion_weights 는 OS 정규화 융합 검색 파이프라인 메타.
    search_backend: str
    opensearch_search_pipeline: str
    opensearch_fusion_weights: tuple[float, float]
    # 023: OS 검색 적합도 컷오프(probe 게이트) 설정. 모두 선택 필드(021 동형) — 미설정 시 측정 근거
    # 기본값. 게이트는 OS 백엔드(이미 opt-in)에만 적용되므로 enabled 기본 True 라도 search_backend='pg'
    # (기본)면 OS 경로 자체가 실행되지 않아 동작 불변(SC-001). eps=상대신호(top-mean) 하한·floor=코사인
    # 절대 backstop·probe_k=probe 표본 수. 범위 밖 값은 _resolve_* 헬퍼가 _build_settings 시점에 즉시
    # ValueError 로 차단한다(잘못된 임계로 검색하지 않게 — _resolve_search_backend fail-fast 동형).
    search_os_cutoff_enabled: bool
    search_os_cutoff_eps: float
    search_os_cutoff_floor: float
    search_os_probe_k: int
    # 024: OS 검색 per-result 적합도 컷오프(정규화 점수 스케일). PG 코사인 스케일 search_min_scores 와
    # 분리 — OS 경로(backend=opensearch)만 적용해 023 버킷 게이트가 유지한 버킷의 노이즈 꼬리를 거른다.
    # 범위 밖은 resolve_os_search_min_scores 가 _build_settings 시점에 즉시 ValueError(fail-fast).
    search_os_min_scores: dict[str, float]


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


def _env_bool_default(name: str, default: bool) -> bool:
    """불리언 선택 환경변수(미설정 시 기본값). ``_require_env_bool`` 의 선택 필드 판본(020)."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    low = str(raw).strip().lower()
    if low in {"1", "true", "yes", "y", "on"}:
        return True
    if low in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"불리언 환경변수 형식 오류: {name}={raw!r}")


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


# 024: OS 검색 per-result 적합도 컷오프 기본값(정규화 융합 점수 스케일). 위 SEARCH_MIN_SCORE_*(PG 코사인
# 스케일)와 **분리** — OS 정규화 점수는 결과셋 내 min-max(top≈1.0)라 스케일이 달라 PG값(image 0.25)이
# 무력이다. G3 calibration 분포(관련 ≥~0.45 군집 ↔ 노이즈 꼬리 ≤~0.35)의 절벽 기준값.
_OS_MIN_SCORE_DEFAULTS: dict[str, float] = {"text": 0.45, "image": 0.45, "video": 0.45, "audio": 0.40}


def resolve_os_search_min_scores() -> dict[str, float]:
    """모달리티 → OS 검색 per-result 적합도 하한(정규화 점수 스케일, 024). ``SEARCH_OS_MIN_SCORE_<M>``
    가 있으면 덮고, 없으면 ``_OS_MIN_SCORE_DEFAULTS[m]``. 범위 ``[0,1]`` 밖이면 **즉시 ValueError**
    (잘못된 임계로 검색하지 않게 — ``_resolve_search_backend`` fail-fast 동형). OS 경로 전용이라
    PG ``SEARCH_MIN_SCORE_*`` 와 독립(스케일 분리·pg 무변경)."""
    out: dict[str, float] = {}
    for m in _SEARCH_MIN_SCORE_MODALITIES:
        value = _env_float_default(f"SEARCH_OS_MIN_SCORE_{m.upper()}", _OS_MIN_SCORE_DEFAULTS[m])
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"OS min_score 범위 오류: SEARCH_OS_MIN_SCORE_{m.upper()}={value!r} (0<=v<=1)")
        out[m] = value
    return out


# 021: 검색 read path 백엔드 화이트리스트. 'pg'=현 media_search 경로(기본·회귀 0), 'opensearch'=020 OS
# 인덱스 하이브리드. 그 외 값은 잘못된 백엔드로 검색하지 않도록 _resolve_search_backend 가 즉시 차단한다.
_SEARCH_BACKENDS = ("pg", "opensearch")


def _resolve_search_backend() -> str:
    """검색 read path 백엔드(021, FR-010). ``SEARCH_BACKEND`` 미설정 시 기본 ``'pg'``(현 경로·동작 불변).

    화이트리스트 ``{pg, opensearch}`` 밖 값은 **즉시 ValueError** 로 차단한다 — 오설정이 런타임까지
    숨지 않게(fail-fast, 백로그 '설정 fail-late' 교정). 019 chunk_agg(헬퍼 지연 검증)와 달리 검증을
    ``_build_settings`` 시점에 끌어와 프로세스 시작 시 빠르게 실패시킨다(잘못된 백엔드 선택 차단)."""
    value = _env_str_default("SEARCH_BACKEND", "pg")
    if value not in _SEARCH_BACKENDS:
        raise ValueError(
            f"지원하지 않는 검색 백엔드: SEARCH_BACKEND={value!r} (지원: {list(_SEARCH_BACKENDS)})"
        )
    return value


def _resolve_opensearch_fusion_weights() -> tuple[float, float]:
    """OS 정규화 융합 가중치 (BM25, kNN)(021, FR-005). ``OPENSEARCH_FUSION_WEIGHTS="0.5,0.5"`` → 튜플.

    미설정 시 기본 ``(0.5, 0.5)``(측정 근거 균형값). build_search_body 의 hybrid 서브쿼리 순서
    ``[텍스트(BM25), knn]`` 와 동일 순서의 ``(w_bm25, w_knn)`` 이다. 정확히 2개(BM25·kNN)가 아니거나,
    수치가 아니거나, 각 가중치가 ``0<=w<=1`` 범위를 벗어나면 **즉시 ValueError**(fail-fast — 잘못된
    융합 가중치로 검색 품질이 조용히 붕괴하지 않게)."""
    raw = os.getenv("OPENSEARCH_FUSION_WEIGHTS")
    if raw is None or not raw.strip():
        return (0.5, 0.5)
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"융합 가중치 개수 오류: OPENSEARCH_FUSION_WEIGHTS={raw!r} (BM25,kNN 두 값 필요)"
        )
    try:
        w_bm25, w_knn = (float(parts[0]), float(parts[1]))
    except ValueError as e:
        raise ValueError(f"융합 가중치 형식 오류: OPENSEARCH_FUSION_WEIGHTS={raw!r}") from e
    for w in (w_bm25, w_knn):
        if not (0.0 <= w <= 1.0):
            raise ValueError(
                f"융합 가중치 범위 오류: OPENSEARCH_FUSION_WEIGHTS={raw!r} (각 0<=w<=1)"
            )
    return (w_bm25, w_knn)


# 023: OS 검색 적합도 컷오프(probe 게이트) 4종 해소. 021 _resolve_opensearch_fusion_weights 와 동형으로
# 환경 파싱·범위 검증·fail-fast 를 _build_settings 시점에 끌어와, 잘못된 임계로 검색하지 않도록 프로세스
# 시작 시 즉시 실패시킨다. enabled 기본 True 라도 search_backend='pg'(기본)면 OS 경로 미실행 → 동작 불변.
def _resolve_os_cutoff_enabled() -> bool:
    """OS 검색 컷오프 활성 여부(023, FR-001). ``SEARCH_OS_CUTOFF_ENABLED`` 미설정 시 기본 ``True``.

    bool 파싱은 020 ``_env_bool_default`` 재사용. 게이트는 OS 백엔드(이미 opt-in)에만 적용되므로
    기본 True 라도 ``search_backend='pg'``(기본)면 OS 경로 자체가 실행되지 않아 동작 불변(SC-001)."""
    return _env_bool_default("SEARCH_OS_CUTOFF_ENABLED", True)


def _resolve_os_cutoff_eps() -> float:
    """컷오프 상대신호 하한 eps(023, FR-001). 기본 ``0.15``(G4 calibration 확정). 범위 ``[0,1)`` 밖이면 **즉시 ValueError**.

    eps 는 ``top − mean``(코사인 차) 의 하한이라 0 이상·1 미만이어야 의미가 있다(코사인 차의 유효 폭).
    범위 밖은 잘못된 임계로 검색하지 않도록 fail-fast(``_resolve_search_backend`` 동형)."""
    value = _env_float_default("SEARCH_OS_CUTOFF_EPS", 0.15)
    if not (0.0 <= value < 1.0):
        raise ValueError(f"컷오프 eps 범위 오류: SEARCH_OS_CUTOFF_EPS={value!r} (0<=eps<1)")
    return value


def _resolve_os_cutoff_floor() -> float:
    """컷오프 절대 backstop floor(023, FR-004). 기본 ``0.43``(G4 calibration 확정). 범위 ``[-1,1]`` 밖이면 **즉시 ValueError**.

    floor 는 top 코사인의 절대 하한이라 코사인 정의역 ``[-1,1]`` 안이어야 한다(경계 포함). 범위 밖은
    잘못된 임계로 검색하지 않도록 fail-fast. 0.43 = 실OS 검증 라벨셋의 있음 top 최소(0.490)·없음 top
    최대(0.365) 갭 중앙(주분리축)."""
    value = _env_float_default("SEARCH_OS_CUTOFF_FLOOR", 0.43)
    if not (-1.0 <= value <= 1.0):
        raise ValueError(f"컷오프 floor 범위 오류: SEARCH_OS_CUTOFF_FLOOR={value!r} (-1<=floor<=1)")
    return value


def _resolve_os_probe_k() -> int:
    """probe 표본 수 probe_k(023, FR-002). 기본 ``50``. ``1`` 미만이면 **즉시 ValueError**.

    probe_k 는 baseline(표본 평균) 추정의 표본 크기라 최소 1 이상이어야 한다(0·음수는 무의미). 정수
    형식 오류는 ``_env_int_default`` 가, 범위(>=1)는 여기서 fail-fast 로 차단한다."""
    value = _env_int_default("SEARCH_OS_PROBE_K", 50)
    if value < 1:
        raise ValueError(f"컷오프 probe_k 범위 오류: SEARCH_OS_PROBE_K={value!r} (probe_k>=1)")
    return value


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
        # 019: 집계 방식의 유효성 검증(미지원 값 차단)은 chunk_agg_config 헬퍼가 수행한다 —
        # 018 active_embed_channel(필드는 raw 저장) + active_embed_model(헬퍼가 ValueError) 과 동형.
        chunk_agg=_env_str_default("SEARCH_CHUNK_AGG", "max"),
        chunk_agg_k=_env_int_default("SEARCH_CHUNK_AGG_K", 3),
        chunk_agg_mix_w=_env_float_default("SEARCH_CHUNK_AGG_MIX_W", 0.5),
        # 020: OpenSearch 동기화 선택 설정. 미설정 시 url/index 기본값·sync off(기존 동작 불변).
        opensearch_url=_env_str_default("OPENSEARCH_URL", "http://localhost:9200"),
        opensearch_index=_env_str_default("OPENSEARCH_INDEX", "assets"),
        opensearch_sync_enabled=_env_bool_default("OPENSEARCH_SYNC_ENABLED", False),
        # 021: 검색 백엔드 선택. 미설정 시 기본 pg(현 경로·회귀 0). search_backend 화이트리스트 검증·
        # 융합 가중치 범위검증은 _resolve_* 헬퍼가 _build_settings 시점에 수행한다(fail-fast).
        search_backend=_resolve_search_backend(),
        opensearch_search_pipeline=_env_str_default("OPENSEARCH_SEARCH_PIPELINE", "assets-hybrid"),
        opensearch_fusion_weights=_resolve_opensearch_fusion_weights(),
        # 023: OS 검색 컷오프 4종. 범위 검증·fail-fast 는 _resolve_* 헬퍼가 _build_settings 시점에 수행
        # (021 fusion_weights 동형). 기본 enabled=True 라도 pg 백엔드(기본)면 OS 경로 미실행 → 회귀 0.
        search_os_cutoff_enabled=_resolve_os_cutoff_enabled(),
        search_os_cutoff_eps=_resolve_os_cutoff_eps(),
        search_os_cutoff_floor=_resolve_os_cutoff_floor(),
        search_os_probe_k=_resolve_os_probe_k(),
        # 024: OS per-result 적합도 컷오프(정규화 스케일). PG search_min_scores 와 분리(OS 경로 전용).
        search_os_min_scores=resolve_os_search_min_scores(),
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


# 019: 지원하는 per-asset 청크 집계 방식. 'max'=기존 MAX(회귀 0), 'topk_mean'=상위 k 평균,
# 'mix'=w*MAX+(1-w)*AVG. 미지원 값은 잘못된 산식으로 검색하지 않도록 chunk_agg_config 가 차단한다.
_CHUNK_AGG_MODES = ("max", "topk_mean", "mix")


@dataclass(frozen=True)
class ChunkAggConfig:
    """per-asset 청크 집계 설정의 네임드 단일 출처(019). 검색 경로가 이 값으로 집계식을 고른다.

    frozen=True 라 동일 설정 → 동일 값(==)이 결정적으로 보장된다(헌법 3조)."""

    agg: str        # 'max' | 'topk_mean' | 'mix'
    k: int          # topk_mean 상위 k
    mix_w: float    # mix 가중치 w


def chunk_agg_config(settings: PipelineSettings | None = None) -> ChunkAggConfig:
    """per-asset 청크 집계 설정(019). 검색 경로(ST 하이브리드·시각 2단계)가 공유하는 단일 출처.

    018 ``active_embed_channel``/``active_embed_model`` 동형 — ``settings`` 미지정 시 활성 설정을
    사용하고(테스트는 ``settings`` 주입으로 순수 단위 검증), 지원하지 않는 ``SEARCH_CHUNK_AGG`` 값은
    즉시 ``ValueError`` 로 차단한다(잘못된 산식으로 검색 방지).

    회귀 0(SC-001): 기본 ``agg='max'`` 는 기존 MAX 집계와 동치다. settings 미초기화(순수 단위 등)에서는
    활성 해소가 ``RuntimeError`` 이므로 기존 MAX 동작을 보존하도록 ``max`` 기본 집계로 보수 폴백한다
    (운영 진입점은 항상 ``init_settings`` 하므로 이 폴백은 비운영 경로 — 오설정을 warning 으로 남긴다)."""
    try:
        cfg = settings if settings is not None else get_current_settings()
    except RuntimeError:
        _LOG.warning("settings 미초기화 — 청크 집계 'max' 보수 폴백(운영은 init_settings 필수)")
        return ChunkAggConfig(agg="max", k=3, mix_w=0.5)
    if cfg.chunk_agg not in _CHUNK_AGG_MODES:
        raise ValueError(
            f"지원하지 않는 청크 집계 방식: {cfg.chunk_agg!r} (지원: {list(_CHUNK_AGG_MODES)})"
        )
    return ChunkAggConfig(agg=cfg.chunk_agg, k=cfg.chunk_agg_k, mix_w=cfg.chunk_agg_mix_w)
