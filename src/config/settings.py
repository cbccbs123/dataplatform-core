"""``.env.*`` 파이프라인 설정(임베딩, CLIP 라벨 상한 등). ``init_settings`` 후 ``get_current_settings`` 로 조회."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from src.config import keyframe_dedup_defaults as _kf
from src.config import search_constants

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedConfig:
    """임베딩 채널·모델·API 백엔드·청킹 설정(017/018/062/063/069 D8)."""

    model: str                      # 활성 로컬 ST 모델(기본 KoSimCSE)
    model_bge: str                  # 017 A/B: BGE-M3 채널('st_bge')용 — 미설정 시 'BAAI/bge-m3'
    active_channel: str             # 018: 운영 활성 채널(적재·검색·관계 단일 출처·기본 'st')
    # 062: API 텍스트 임베딩 백엔드(온프레미스 bge-m3 서빙·채널 'st_api'). 기본 로컬이라 미설정 무영향 —
    # 'st_api' 활성화 시에만 참조. base_url 미설정+active=st_api 는 _validate 가 기동 시점에 차단.
    api_base_url: str
    api_model: str
    api_key: str
    api_timeout_s: float
    api_batch_size: int
    api_max_retries: int
    # 063: image/video CLIP 시각 임베딩 생성 토글. 기본 True=동작 불변(회귀 0). off 면 clip 아이템만 스킵.
    enable_clip: bool
    chunk_size: int                 # 임베딩 청킹 크기(요약용 chunk_size 와 별개)
    # 069 D8: 임베딩 청크 overlap(기본 0=현행·동작 불변). 0<=overlap<chunk_size 는 _validate 가 fail-fast.
    chunk_overlap: int
    normalize: bool


@dataclass(frozen=True)
class SearchConfig:
    """검색 read path 튜닝(021/023/024/025/027/028/029/044/045/073/074/075). OS 단일 백엔드(037)."""

    # 021/037: 검색 백엔드. 037 에서 PG 경로 제거로 'opensearch' 전용 — 화이트리스트 밖은 즉시 ValueError.
    backend: str
    fusion_weights: tuple[float, float]  # 027 클라이언트 융합 (BM25,kNN) 가중치·각 0<=w<=1
    # 023/027: OS 버킷 게이트(robust baseline). eps=상대신호(top-baseline) 하한·floor=코사인 절대 backstop.
    os_cutoff_enabled: bool
    os_cutoff_eps: float
    os_cutoff_floor: float
    # 027: per-result 컷 코사인 하한(024 정규화 스케일 4종을 단일 코사인 임계로 대체). 행 유지=BM25 OR 코사인≥이값.
    os_result_floor: float
    # 028: reranker 평가(쌍별 절대 판정·추론만). 기본 off. τ∈[0,1]·R≥1 fail-fast.
    os_rerank_enabled: bool
    os_rerank_model: str
    os_rerank_top_r: int
    os_rerank_tau: float
    # 029/075: 질의 정규화 토글·방식("morph"=nori 072 기본 | "llm"=gemma 029).
    os_query_norm_enabled: bool
    os_query_norm_method: str
    # 073: aboutness OR-증거 필터(기본 off). 질의가 명사구라 가정 → query-norm on 과 함께 켜는 전제.
    about_filter_enabled: bool
    # 074: 검색시점 top-3 개별 LLM 검증(기본 off). 자연어(어절≥3) 상위 3 자산 gemma 판정.
    llm_verify_enabled: bool
    # 025: OS BM25 operator(기본 'or'·현행). 'and'=전 토큰 매칭(복합어 가짜매칭 F2 차단). 화이트리스트 fail-fast.
    os_bm25_operator: str
    # 044: 필드 evidence 기반 lexical rescue 게이트(런타임 on/off·관측). 임계·가중은 search_constants.
    evidence_rescue_enabled: bool
    evidence_debug: bool            # 044: per-hit debug meta opt-in(keep_reason·matched_queries)


@dataclass(frozen=True)
class OpenSearchConfig:
    """OpenSearch 인프라(색인 대상·동기화·색인 빌더 교정). 020/026/038. 검색 튜닝(SearchConfig)과 직교."""

    url: str
    index: str
    # 038: sync 기본 True — 037 로 검색이 OS 단일이 된 뒤 적재 시 증분 색인이 꺼지면 신규 자산이 검색에서
    # 누락(PG 폴백 없음). 정합 가드(_validate)가 backend=opensearch ∧ ¬sync 를 빌드 시점에 차단 — 기본값·가드 한 쌍.
    sync_enabled: bool
    # 026: 커스텀 nori analyzer user_dictionary 외래어 목록(분해 방지). 빈 항목은 resolver 가 fail-fast.
    nori_user_words: tuple[str, ...]
    # 026: 파일명 정제 추가 잡음 regex(기본 빈). 컴파일 불가·빈 항목은 resolver 가 fail-fast.
    filename_noise_patterns: tuple[str, ...]


@dataclass(frozen=True)
class RelationsConfig:
    """cross-asset 관계 제안 게이트(032/033/008/009)."""

    top_k: int
    min_sim: float                  # 후보 코사인 유사도 하한 — 이 미만은 LLM 에 넣지 않음(기본 0.2)
    auto_approve_min: float         # 이 이상이면 HITL 없이 자동 승인(기본 0.9)
    # 033 FR-002: 자동승인 emb_score 하한(기본 0.0=무력·현 동작). AND 게이트(conf AND emb)에서 0 이면 conf 단독.
    auto_approve_emb_min: float
    # 008 C-2: 경로 신호(path_signal) 후보 별도 한도(동일 폴더 폭주 차단). union ≤ top_k + path_top_k.
    path_top_k: int
    # 009: 관계 재시도 큐 재시도 상한(attempts 도달 시 failed/DLQ 승격). 기본 3.
    retry_max_attempts: int
    # ── 081 승인·노출 게이트 (전부 끌 수 있다 · `src/relations/approval_policy.py` 가 해석) ──
    # 유사도 계열(duplicate_near·same_domain) 저신뢰 제안을 **행으로 만들지 않는** 하한. 0=끔.
    persist_min_conf_similarity: float
    # 신뢰도와 무관하게 자동승인에서 제외할 종류(쉼표 구분). ""=제외 없음(기존 동작).
    auto_approve_exclude_kinds: str
    # 사람 검토 큐에서 뺄 종류(쉼표 구분 · 삭제 아닌 필터). ""=전건 검토(기존 동작).
    review_exempt_kinds: str


@dataclass(frozen=True)
class VideoConfig:
    """영상 키프레임 추출·near-dup 제거(048, FR-501). dedup_* 7필드는 KeyframeDedupConfig 단일 출처."""

    max_keyframes: int
    dedup_enabled: bool             # 기본 True(2026-06-29 결정). off 경로는 추출 바이트 동일(회귀 안전판).
    dedup_hash_max: int
    dedup_ssim_min: float
    dedup_ssim_gray_lo: float
    dedup_hist_min: float
    dedup_compare_mode: str         # 'global' 은 타임라인 손실 위험이라 비기본(SC-008)
    dedup_recent_window: int
    labels_meta_top_k: int          # 영상 키프레임 CLIP 라벨 메타 상한


@dataclass(frozen=True)
class VlmConfig:
    """VLM 요약(049)·이미지 CLIP 라벨 상한/임계. 비전 처리 계열."""

    # 049: VLM 요약 프롬프트 v2 토글(기본 False=v1 바이트 동일·회귀 안전판·FR-102).
    summary_prompt_v2: bool
    summary_ab_judge: bool          # A/B 측정 하니스 LLM-judge 옵션(평가용·추출 무영향)
    image_labels_meta_top_k: int    # 이미지 CLIP 라벨 메타 상한
    labels_score_min: float         # image/video 공통 CLIP 라벨 점수 하한


@dataclass(frozen=True)
class TopicConfig:
    """관계 topic 정본화(058)."""

    # 058: canonicalize 배선 토글(기본 False=동작 불변·시드 전 동치). 빈 레지스트리에서 켜면 raw topic
    # 자동등록 부작용이라 기본 off; 시드(G5) 후 명시 활성화.
    canonicalize_enabled: bool


@dataclass(frozen=True)
class PipelineSettings:
    """파이프라인 실행 설정. ``init_settings(profile)`` 이 한 번만 생성하며 이후 변경 불가(frozen).

    069 US-E FR-E4(PR4b): 64개 평평 필드를 도메인별 하위 frozen dataclass로 묶었다. 도메인 무관 전역만
    상위에 남기고(프로파일·LLM 접속·인코딩·요약/청킹 공통), 나머지는 ``embed``/``search``/``opensearch``/
    ``relations``/``video``/``vlm``/``topic`` 로 접근한다(예: ``cfg.search.os_cutoff_eps``). 필드 조립은
    ``_FIELD_SPECS`` 단일 출처(그룹 컬럼)에서 파생한다.
    """

    profile: Literal["dev", "prod"]
    # 상위 공통(도메인 무관 전역): LLM 접속·프로파일·인코딩·텍스트 추출/요약 공통 파라미터.
    meta_model: str
    openai_base_url: str
    openai_api_key: str
    encoding: str
    summary_max_chars: int
    top_k_keywords: int
    chunk_size: int                 # 요약용 청킹(임베딩 청킹은 embed.chunk_size 로 별개)
    overlap_size: int
    # 도메인 하위 묶음.
    embed: EmbedConfig
    search: SearchConfig
    opensearch: OpenSearchConfig
    relations: RelationsConfig
    video: VideoConfig
    vlm: VlmConfig
    topic: TopicConfig


_SETTINGS: PipelineSettings | None = None


def _require_env(name: str) -> str:
    """**필수** 환경변수를 읽는다 — 없거나 비어 있으면 기동을 멈춘다.

    설정 누락을 기본값으로 덮으면 어떤 값으로 돌고 있는지 아무도 모르게 되므로,
    필수 항목은 조용히 넘기지 않고 즉시 실패시킨다.

    Args:
        name: 환경변수 이름.

    Returns:
        앞뒤 공백을 자른 값.

    Raises:
        ValueError: 미설정이거나 공백뿐일 때.
    """
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"필수 환경변수 누락: {name}")
    return value.strip()


def _require_env_int(name: str) -> int:
    """필수 환경변수를 정수로 읽는다.

    Args:
        name: 환경변수 이름.

    Returns:
        정수 값.

    Raises:
        ValueError: 미설정이거나 정수로 바꿀 수 없을 때(원본 값을 메시지에 담는다).
    """
    raw = _require_env(name)
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"정수 환경변수 형식 오류: {name}={raw!r}") from e


def _require_env_bool(name: str) -> bool:
    """필수 환경변수를 불리언으로 읽는다.

    참으로 보는 값: ``1``·``true``·``yes``·``y``·``on`` / 거짓: ``0``·``false``·``no``·``n``·``off``.
    **그 밖의 값은 예외**다 — 오타를 False 로 조용히 해석하면 기능이 꺼진 줄 모르게 된다.

    Args:
        name: 환경변수 이름.

    Returns:
        불리언 값.

    Raises:
        ValueError: 미설정이거나 인식할 수 없는 표기일 때.
    """
    raw = _require_env(name).lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"불리언 환경변수 형식 오류: {name}={raw!r}")


def _env_int_default(name: str, default: int) -> int:
    """선택 환경변수를 정수로 읽는다(미설정이면 기본값).

    Args:
        name: 환경변수 이름.
        default: 미설정·빈 값일 때 쓸 값.

    Returns:
        정수 값 또는 기본값.

    Raises:
        ValueError: 값이 있는데 정수가 아닐 때 — **기본값으로 넘기지 않는다**(오타를
            숨기면 의도와 다른 설정으로 운영된다).
    """
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError as e:
        raise ValueError(f"정수 환경변수 형식 오류: {name}={raw!r}") from e


def _env_str_default(name: str, default: str) -> str:
    """선택 환경변수를 문자열로 읽는다(미설정·공백뿐이면 기본값).

    Args:
        name: 환경변수 이름.
        default: 미설정일 때 쓸 값.

    Returns:
        앞뒤 공백을 자른 값 또는 기본값.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_float_default(name: str, default: float) -> float:
    """선택 환경변수를 실수로 읽는다(미설정이면 기본값).

    Args:
        name: 환경변수 이름.
        default: 미설정·빈 값일 때 쓸 값.

    Returns:
        실수 값 또는 기본값.

    Raises:
        ValueError: 값이 있는데 실수가 아닐 때(조용히 기본값으로 넘기지 않는다).
    """
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError as e:
        raise ValueError(f"실수 환경변수 형식 오류: {name}={raw!r}") from e


def _env_bool_default(name: str, default: bool) -> bool:
    """선택 환경변수를 불리언으로 읽는다(미설정이면 기본값).

    표기 규칙은 필수판과 같고, **인식 못 하는 값은 예외**다(기본값으로 흡수하지 않는다).

    Args:
        name: 환경변수 이름.
        default: 미설정·빈 값일 때 쓸 값.

    Returns:
        불리언 값 또는 기본값.

    Raises:
        ValueError: 값이 있는데 인식할 수 없는 표기일 때.
    """
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    low = str(raw).strip().lower()
    if low in {"1", "true", "yes", "y", "on"}:
        return True
    if low in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"불리언 환경변수 형식 오류: {name}={raw!r}")


# 037 OpenSearch 전용 정리: 검색 read path 는 020 OS 인덱스 하이브리드 단일 경로다. 021 의 'pg'
# (media_search FTS/벡터) 분기는 제거됐으므로 화이트리스트도 'opensearch' 하나만 남긴다 — 과거 기본값
# 'pg' 를 포함한 그 외 값은 잘못된 백엔드로 검색하지 않도록 _resolve_search_backend 가 즉시 차단한다.
_SEARCH_BACKENDS = ("opensearch",)


def _resolve_search_backend() -> str:
    """검색 read path 백엔드(021, FR-010·037 전환). ``SEARCH_BACKEND`` 미설정 시 기본 ``'opensearch'``.

    037 에서 PG 검색 경로를 제거하며 기본값을 'pg'→'opensearch' 로 전환했다. 화이트리스트
    ``{opensearch}`` 밖 값(과거 'pg' 포함)은 **즉시 ValueError** 로 차단한다 — 오설정이 런타임까지
    숨지 않게(fail-fast, 백로그 '설정 fail-late' 교정). 헬퍼 지연 검증(선택 설정)과 달리 검증을
    ``_build_settings`` 시점에 끌어와 프로세스 시작 시 빠르게 실패시킨다(잘못된 백엔드 선택 차단)."""
    value = _env_str_default("SEARCH_BACKEND", "opensearch")
    if value not in _SEARCH_BACKENDS:
        raise ValueError(
            f"지원하지 않는 검색 백엔드: SEARCH_BACKEND={value!r} (지원: {list(_SEARCH_BACKENDS)})"
        )
    return value


def _validate_settings_consistency(settings: PipelineSettings) -> None:
    """교차필드 정합 fail-fast(038 — 037 불변식 'OS read ⇒ OS write 필수').

    검색을 OpenSearch 에서 읽는데(``search_backend=='opensearch'``) 적재 시 OS 증분 색인이 꺼져 있으면
    (``¬opensearch_sync_enabled``) 신규·변경 자산이 검색에서 조용히 누락된다(037 로 PG 폴백 없음).
    이 반쪽 마이그레이션을 ``_build_settings`` 시점에 즉시 차단한다 — 단일 필드 화이트리스트를 보는
    ``_resolve_search_backend`` 와 달리 두 필드의 결합 불변식이라 빌드 완료 후 검증한다(런타임까지
    오설정이 숨지 않게). 037 로 ``search_backend`` 가 'opensearch' 단일이므로 사실상 'sync 는 항상 켜야
    한다'와 같지만, 결합 형태로 적어 의도(왜 켜야 하나)를 드러내고 향후 백엔드 추가에도 견고하게 둔다."""
    if settings.search.backend == "opensearch" and not settings.opensearch.sync_enabled:
        raise ValueError(
            "설정 불일치: SEARCH_BACKEND=opensearch 인데 OPENSEARCH_SYNC_ENABLED=false 입니다. "
            "OS 검색은 적재 시 OS 증분 색인이 필수입니다(037 이후 PG 폴백 없음 — 끄면 신규 자산이 "
            "검색에서 누락). OPENSEARCH_SYNC_ENABLED=true 로 설정하세요."
        )
    # E6: 활성 임베딩 채널은 지원 목록(_TEXT_EMBED_CHANNELS = model_for_channel 매핑 키) 안이어야 한다.
    #   미검증이면 오타 채널이 빌드를 통과하고, model_for_channel 이 실제 호출되는 파이프라인 한복판
    #   (적재·검색)에서야 ValueError 로 터진다(fail-late). 038/062 관례대로 기동 시점에 즉시 차단한다.
    if settings.embed.active_channel not in _TEXT_EMBED_CHANNELS:
        raise ValueError(
            f"설정 불일치: EMBED_ACTIVE_CHANNEL={settings.embed.active_channel!r} 은(는) 지원하지 않는 "
            f"임베딩 채널입니다 (지원: {sorted(_TEXT_EMBED_CHANNELS)})."
        )
    # 062: API 임베딩 채널(st_api) 활성인데 base_url 미설정이면 파이프라인 한복판(/embeddings 호출)이
    #   아니라 기동 시점에 즉시 차단한다(038 fail-fast 관례와 통일 — 채널만 켜는 사람 실수 방지).
    if backend_for_channel(settings.embed.active_channel, settings) == "api" and not settings.embed.api_base_url:
        raise ValueError(
            "설정 불일치: EMBED_ACTIVE_CHANNEL=st_api(API 임베딩) 인데 EMBED_API_BASE_URL 이 비어 있습니다. "
            "API 백엔드는 엔드포인트 주입이 필수입니다 — EMBED_API_BASE_URL 을 설정하세요(예: http://<host>:<port>/v1)."
        )
    # 069 D8: 임베딩 청크 overlap 은 iter_document_chunks 가 0<=overlap<chunk_size 를 요구한다. 위반 시
    #   첫 문서 처리 시점(파이프라인 한복판)에야 ValueError 가 터지므로, opt-in 오설정을 기동 시점에
    #   즉시 차단한다(038/062 fail-fast 관례와 통일). 기본 0 은 항상 통과(동작 불변).
    if not (0 <= settings.embed.chunk_overlap < settings.embed.chunk_size):
        raise ValueError(
            f"설정 불일치: TEXT_EMBED_CHUNK_OVERLAP={settings.embed.chunk_overlap} 는 "
            f"0 이상이고 TEXT_EMBED_CHUNK_SIZE={settings.embed.chunk_size} 미만이어야 합니다 "
            "(청크 겹침은 청크 크기보다 작아야 함 — iter_document_chunks 계약)."
        )


def _resolve_opensearch_fusion_weights() -> tuple[float, float]:
    """OS 클라이언트 융합 가중치 (BM25, kNN)(021·027, FR-005). ``OPENSEARCH_FUSION_WEIGHTS="0.5,0.5"`` → 튜플.

    미설정 시 기본은 ``search_constants.OS_FUSION_WEIGHTS_DEFAULT`` 단일 출처(F1 — 하드코딩 제거).
    fuse_hybrid 의 서브검색 순서 ``[BM25, kNN]`` 과 동일 순서의 ``(w_bm25, w_knn)`` 이다. 정확히
    2개(BM25·kNN)가 아니거나, 수치가 아니거나, 각 가중치가 ``0<=w<=1`` 범위를 벗어나면 **즉시
    ValueError**(fail-fast — 잘못된 융합 가중치로 검색 품질이 조용히 붕괴하지 않게)."""
    raw = os.getenv("OPENSEARCH_FUSION_WEIGHTS")
    if raw is None or not raw.strip():
        return search_constants.OS_FUSION_WEIGHTS_DEFAULT
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


# 023/027: OS 검색 버킷 게이트 + per-result 컷 설정 해소. 021 _resolve_opensearch_fusion_weights 와
# 동형으로 환경 파싱·범위 검증·fail-fast 를 _build_settings 시점에 끌어와, 잘못된 임계로 검색하지 않도록
# 프로세스 시작 시 즉시 실패시킨다. 기본값은 모두 search_constants 단일 출처(F1 — 하드코딩 제거).
# 게이트는 OS 검색 경로(037 단일 백엔드)에 적용된다.
def _resolve_os_cutoff_enabled() -> bool:
    """OS 검색 게이트 활성 여부(023·027, FR-001). 미설정 시 기본 ``OS_CUTOFF_ENABLED_DEFAULT``(True).

    bool 파싱은 020 ``_env_bool_default`` 재사용. 게이트는 OS 검색 경로(037 단일 백엔드)에 적용된다."""
    return _env_bool_default("SEARCH_OS_CUTOFF_ENABLED", search_constants.OS_CUTOFF_ENABLED_DEFAULT)


def _resolve_os_cutoff_eps() -> float:
    """게이트 상대신호 하한 eps(023·027, FR-001). 기본 ``OS_CUTOFF_EPS_DEFAULT``. 범위 ``[0,1)`` 밖이면 **즉시 ValueError**.

    eps 는 ``top − baseline``(코사인 차) 의 하한이라 0 이상·1 미만이어야 의미가 있다(코사인 차의 유효 폭).
    범위 밖은 잘못된 임계로 검색하지 않도록 fail-fast(``_resolve_search_backend`` 동형). 027: baseline
    정의가 '하위 절반 평균'(robust)으로 바뀌어 실측 확정치는 search_constants 1곳만 갱신한다."""
    value = _env_float_default("SEARCH_OS_CUTOFF_EPS", search_constants.OS_CUTOFF_EPS_DEFAULT)
    if not (0.0 <= value < 1.0):
        raise ValueError(f"컷오프 eps 범위 오류: SEARCH_OS_CUTOFF_EPS={value!r} (0<=eps<1)")
    return value


def _resolve_os_cutoff_floor() -> float:
    """게이트 절대 backstop floor(023·027, FR-004). 기본 ``OS_CUTOFF_FLOOR_DEFAULT``. 범위 ``[-1,1]`` 밖이면 **즉시 ValueError**.

    floor 는 top 코사인의 절대 하한이라 코사인 정의역 ``[-1,1]`` 안이어야 한다(경계 포함). 범위 밖은
    잘못된 임계로 검색하지 않도록 fail-fast. 실측 확정치는 search_constants 1곳만 갱신한다(F1)."""
    value = _env_float_default("SEARCH_OS_CUTOFF_FLOOR", search_constants.OS_CUTOFF_FLOOR_DEFAULT)
    if not (-1.0 <= value <= 1.0):
        raise ValueError(f"컷오프 floor 범위 오류: SEARCH_OS_CUTOFF_FLOOR={value!r} (-1<=floor<=1)")
    return value


def _resolve_os_result_floor() -> float:
    """OS per-result 컷 코사인 하한(027, FR-004). 기본 ``OS_RESULT_FLOOR_DEFAULT``. 범위 ``[-1,1]`` 밖이면 **즉시 ValueError**.

    행 유지 = BM25 매칭(어휘 증거) OR 원시 코사인 ≥ 이 값(의미 증거). 024 의 정규화 스케일 임계 4종을
    대체하는 전역 1개(코사인 스케일). 코사인 정의역 안이어야 하므로 ``[-1,1]`` 밖은 잘못된 임계로 검색
    하지 않도록 fail-fast(``_resolve_os_cutoff_floor`` 동형). 실측 확정치는 search_constants 1곳만 갱신."""
    value = _env_float_default("SEARCH_OS_RESULT_FLOOR", search_constants.OS_RESULT_FLOOR_DEFAULT)
    if not (-1.0 <= value <= 1.0):
        raise ValueError(f"OS result_floor 범위 오류: SEARCH_OS_RESULT_FLOOR={value!r} (-1<=v<=1)")
    return value


# 025: OS BM25 operator 화이트리스트. 'or'=현행(부분 토큰 매칭 허용·회귀 0), 'and'=전 토큰 매칭(F2).
_OS_BM25_OPERATORS = ("or", "and")


_OS_QUERY_NORM_METHODS = ("morph", "llm")


def _resolve_os_query_norm_method() -> str:
    """질의 정규화 방식(075). 미설정 시 ``OS_QUERY_NORM_METHOD_DEFAULT``('morph'·072 채택값).

    화이트리스트 {morph, llm} 밖 값은 **즉시 ValueError**(fail-fast — _resolve_os_bm25_operator 동형)."""
    value = _env_str_default(
        "SEARCH_OS_QUERY_NORM_METHOD", search_constants.OS_QUERY_NORM_METHOD_DEFAULT
    ).lower()
    if value not in _OS_QUERY_NORM_METHODS:
        raise ValueError(
            f"지원하지 않는 질의 정규화 방식: SEARCH_OS_QUERY_NORM_METHOD={value!r} "
            f"(지원: {list(_OS_QUERY_NORM_METHODS)})"
        )
    return value


def _resolve_os_bm25_operator() -> str:
    """OS BM25 multi_match operator(025, FR-001). 미설정 시 ``OS_BM25_OPERATOR_DEFAULT``('or', 현행·F1).

    화이트리스트 {or, and} 밖 값은 **즉시 ValueError**(fail-fast — _resolve_search_backend 동형)."""
    value = _env_str_default("SEARCH_OS_BM25_OPERATOR", search_constants.OS_BM25_OPERATOR_DEFAULT).lower()
    if value not in _OS_BM25_OPERATORS:
        raise ValueError(
            f"지원하지 않는 BM25 operator: SEARCH_OS_BM25_OPERATOR={value!r} (지원: {list(_OS_BM25_OPERATORS)})"
        )
    return value


def resolve_opensearch_nori_user_words() -> tuple[str, ...]:
    """nori user_dictionary 외래어 목록(026 FR-004). ``OPENSEARCH_NORI_USER_WORDS="아이패드,아이폰,..."``
    CSV 로 오버라이드, 미설정 시 기본 목록(``search_constants.NORI_USER_WORDS_DEFAULT`` 단일 출처 —
    069 T302). 외래어가 nori 로 분해되면('아이패드'→'아이'+'패드') 정확매칭 무력화·가짜매칭이 생기므로
    한 토큰으로 보존한다. 공백 항목은 nori 사전 규칙으로 무의미·거부 대상이라 **즉시 ValueError**
    (fail-fast — 잘못된 사전으로 인덱스를 만들지 않게, ``_resolve_search_backend`` 동형)."""
    raw = os.getenv("OPENSEARCH_NORI_USER_WORDS")
    if raw is None or not raw.strip():
        return search_constants.NORI_USER_WORDS_DEFAULT
    words = [w.strip() for w in raw.split(",")]
    if any(not w for w in words):
        raise ValueError(
            f"nori user_dictionary 빈 항목: OPENSEARCH_NORI_USER_WORDS={raw!r} (공백 항목 금지)"
        )
    return tuple(words)


def resolve_opensearch_filename_noise_patterns() -> tuple[str, ...]:
    """파일명 정제 추가 잡음 regex 목록(026 FR-003③). ``OPENSEARCH_FILENAME_NOISE_PATTERNS="re1,re2"``
    CSV, 미설정 시 **빈 목록**(기본 정제는 clean_file_name 의 ID스러움 판정). 공백 항목·컴파일 불가
    패턴은 **즉시 ValueError**(fail-fast — 잘못된 정제로 색인하지 않게). regex 안에 콤마가 필요한
    드문 경우는 본 CSV 로 표현 불가하니 코드에서 직접 주입한다(기본 빈이라 통상 미사용)."""
    raw = os.getenv("OPENSEARCH_FILENAME_NOISE_PATTERNS")
    if raw is None or not raw.strip():
        return ()
    pats = [p.strip() for p in raw.split(",")]
    if any(not p for p in pats):
        raise ValueError(
            f"파일명 잡음 패턴 빈 항목: OPENSEARCH_FILENAME_NOISE_PATTERNS={raw!r} (공백 항목 금지)"
        )
    for p in pats:
        try:
            re.compile(p)
        except re.error as e:
            raise ValueError(
                f"파일명 잡음 패턴 컴파일 오류: OPENSEARCH_FILENAME_NOISE_PATTERNS={raw!r} ({p!r}: {e})"
            ) from e
    return tuple(pats)


def _resolve_os_rerank_top_r() -> int:
    """rerank 후보 상한(028). 기본 search_constants 단일 출처. 1 미만은 즉시 ValueError(fail-fast)."""
    value = _env_int_default("SEARCH_OS_RERANK_TOP_R", search_constants.OS_RERANK_TOP_R_DEFAULT)
    if value < 1:
        raise ValueError(f"rerank top_r 범위 오류: SEARCH_OS_RERANK_TOP_R={value!r} (>=1)")
    return value


def _resolve_os_rerank_tau() -> float:
    """rerank 쌍별 점수 하한 τ(028). 범위 [0,1] 밖은 즉시 ValueError(fail-fast)."""
    value = _env_float_default("SEARCH_OS_RERANK_TAU", search_constants.OS_RERANK_TAU_DEFAULT)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"rerank tau 범위 오류: SEARCH_OS_RERANK_TAU={value!r} (0<=tau<=1)")
    return value


def _opt_int(default: int) -> Callable[[str], int]:
    """기본값을 미리 묶어 "환경변수 키 → 정수" 읽기 함수를 만든다.

    아래 필드 표가 한 줄에 하나씩 선언되도록 하는 장치다 — 표에는 기본값만 적고,
    실제 읽기는 만들어진 함수가 한다.
    """
    return lambda key: _env_int_default(key, default)


def _opt_str(default: str) -> Callable[[str], str]:
    """기본값을 묶어 "키 → 문자열" 읽기 함수를 만든다(``_opt_int`` 와 같은 방식)."""
    return lambda key: _env_str_default(key, default)


def _env_str_allow_empty(name: str, default: str) -> str:
    """선택 환경변수를 문자열로 읽되 **빈 값을 "명시적 빈 값"으로 존중**한다.

    ``_env_str_default`` 는 ``""`` 을 미설정으로 보고 기본값으로 되돌린다 — 인덱스명·모델명처럼
    "빈 값이 무의미한" 설정에는 그게 맞다. 그러나 **목록형 게이트**에서는 정반대다: ``""`` 은
    *"이 게이트를 끈다"* 는 유일한 표현 수단인데 기본값으로 되돌리면 **끌 방법이 사라진다.**
    081 게이트는 롤백이 코드 revert 가 아니라 설정 변경이어야 하므로(관계 전량 재생성이 약
    28시간) 이 구분이 필요하다. 전역 헬퍼를 바꾸지 않고 별 함수를 두는 이유는 기존 문자열
    설정들의 동작을 건드리지 않기 위해서다.

    Args:
        name: 환경변수 이름.
        default: **미설정일 때만** 쓸 값(빈 문자열이 설정된 경우에는 쓰지 않는다).

    Returns:
        미설정이면 ``default``, 설정됐으면 앞뒤 공백을 자른 값(빈 문자열 가능).
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _opt_str_allow_empty(default: str) -> Callable[[str], str]:
    """``_env_str_allow_empty`` 를 기본값과 묶어 필드 표에 한 줄로 쓸 수 있게 한다.

    Args:
        default: 환경변수 미설정 시 쓸 값.

    Returns:
        "키 → 문자열" 읽기 함수.
    """
    return lambda key: _env_str_allow_empty(key, default)


def _opt_float(default: float) -> Callable[[str], float]:
    """기본값을 묶어 "키 → 실수" 읽기 함수를 만든다."""
    return lambda key: _env_float_default(key, default)


def _opt_bool(default: bool) -> Callable[[str], bool]:
    """기본값을 묶어 "키 → 불리언" 읽기 함수를 만든다.

    Args:
        default: 환경변수가 없을 때 쓸 값.

    Returns:
        키를 받아 불리언을 돌려주는 함수(값을 바로 읽는 것이 아니라 **읽는 함수**를 만든다).
    """
    return lambda key: _env_bool_default(key, default)


def _video_max_keyframes(key: str) -> int:
    """영상 키프레임 상한을 읽는다 — **0 이하는 기본값으로 보정**한다.

    0을 그대로 쓰면 키프레임을 하나도 뽑지 않아 영상이 통째로 검색에서 빠진다.
    "제한 없음"을 0으로 표현하려던 입력을 사고로 만들지 않으려는 보정이다.
    """
    vk = _env_int_default(key, 48)
    return 48 if vk <= 0 else vk


class _Spec(NamedTuple):
    """한 필드 = 한 행: 그룹·속성명·env 키·읽기함수·필수여부.

    group="" 는 상위 공통(PipelineSettings 직속), 그 외는 하위 dataclass 이름(embed/search/…).
    attr 은 그룹 내 필드명(공통이면 PipelineSettings 필드명).
    """

    group: str
    attr: str
    env: str
    read: Callable[[str], Any]
    required: bool = False


# 하위 dataclass 이름 → 클래스(빌드 시 그룹별 조립). 상위 공통(group="")은 여기 없음.
_GROUP_CLASSES: dict[str, type] = {
    "embed": EmbedConfig,
    "search": SearchConfig,
    "opensearch": OpenSearchConfig,
    "relations": RelationsConfig,
    "video": VideoConfig,
    "vlm": VlmConfig,
    "topic": TopicConfig,
}


# ── 069 US-E FR-E4: 필드 명세 단일 출처(그룹 포함) ─────────────────────────────
# build 조립(그룹별 하위 dataclass)·테스트 격리키·커버리지가 모두 이 한 테이블에서 파생된다 — 새 필드
# 추가 시 이 테이블 한 행만(SC-E). 위 dataclass 정의와 이 테이블은 test_settings.TestFieldSpecsSSOT 가
# 봉인한다. read 세 종류: 필수(_require_env*·required=True)·선택(_opt_*(기본값))·검증 resolver(_resolve_*·
# env 키 내부 하드코드라 read 는 호출만·spec 의 env 는 격리키 파생용 메타).
_FIELD_SPECS: tuple[_Spec, ...] = (
    # ── 상위 공통(group="") ──
    _Spec("", "meta_model", "META_MODEL", _require_env, required=True),
    _Spec("", "openai_base_url", "OPENAI_BASE_URL", _require_env, required=True),
    _Spec("", "openai_api_key", "OPENAI_API_KEY", _require_env, required=True),
    _Spec("", "encoding", "ENCODING", _require_env, required=True),
    _Spec("", "summary_max_chars", "SUMMARY_MAX_CHARS", _require_env_int, required=True),
    _Spec("", "top_k_keywords", "TOP_K_KEYWORDS", _require_env_int, required=True),
    _Spec("", "chunk_size", "CHUNK_SIZE", _require_env_int, required=True),
    _Spec("", "overlap_size", "OVERLAP_SIZE", _require_env_int, required=True),
    # ── embed ──
    _Spec("embed", "model", "TEXT_EMBED_MODEL", _require_env, required=True),
    _Spec("embed", "chunk_size", "TEXT_EMBED_CHUNK_SIZE", _require_env_int, required=True),
    _Spec("embed", "normalize", "TEXT_EMBED_NORMALIZE", _require_env_bool, required=True),
    _Spec("embed", "model_bge", "TEXT_EMBED_MODEL_BGE", _opt_str("BAAI/bge-m3")),
    _Spec("embed", "active_channel", "EMBED_ACTIVE_CHANNEL", _opt_str("st")),
    _Spec("embed", "api_base_url", "EMBED_API_BASE_URL", _opt_str("")),
    _Spec("embed", "api_model", "EMBED_API_MODEL", _opt_str("BAAI/bge-m3")),
    _Spec("embed", "api_key", "EMBED_API_KEY", _opt_str("")),
    _Spec("embed", "api_timeout_s", "EMBED_API_TIMEOUT_S", _opt_float(30.0)),
    _Spec("embed", "api_batch_size", "EMBED_API_BATCH_SIZE", _opt_int(32)),
    _Spec("embed", "api_max_retries", "EMBED_API_MAX_RETRIES", _opt_int(2)),
    _Spec("embed", "enable_clip", "EMBED_ENABLE_CLIP", _opt_bool(True)),
    _Spec("embed", "chunk_overlap", "TEXT_EMBED_CHUNK_OVERLAP", _opt_int(0)),
    # ── search(튜닝·검증 resolver) ──
    _Spec("search", "backend", "SEARCH_BACKEND", lambda _k: _resolve_search_backend()),
    _Spec("search", "fusion_weights", "OPENSEARCH_FUSION_WEIGHTS", lambda _k: _resolve_opensearch_fusion_weights()),
    _Spec("search", "os_cutoff_enabled", "SEARCH_OS_CUTOFF_ENABLED", lambda _k: _resolve_os_cutoff_enabled()),
    _Spec("search", "os_cutoff_eps", "SEARCH_OS_CUTOFF_EPS", lambda _k: _resolve_os_cutoff_eps()),
    _Spec("search", "os_cutoff_floor", "SEARCH_OS_CUTOFF_FLOOR", lambda _k: _resolve_os_cutoff_floor()),
    _Spec("search", "os_result_floor", "SEARCH_OS_RESULT_FLOOR", lambda _k: _resolve_os_result_floor()),
    _Spec("search", "os_rerank_enabled", "SEARCH_OS_RERANK_ENABLED", _opt_bool(search_constants.OS_RERANK_ENABLED_DEFAULT)),
    _Spec("search", "os_rerank_model", "SEARCH_OS_RERANK_MODEL", _opt_str(search_constants.OS_RERANK_MODEL_DEFAULT)),
    _Spec("search", "os_rerank_top_r", "SEARCH_OS_RERANK_TOP_R", lambda _k: _resolve_os_rerank_top_r()),
    _Spec("search", "os_rerank_tau", "SEARCH_OS_RERANK_TAU", lambda _k: _resolve_os_rerank_tau()),
    _Spec("search", "os_query_norm_enabled", "SEARCH_OS_QUERY_NORM_ENABLED", _opt_bool(search_constants.OS_QUERY_NORM_ENABLED_DEFAULT)),
    _Spec("search", "os_query_norm_method", "SEARCH_OS_QUERY_NORM_METHOD", lambda _k: _resolve_os_query_norm_method()),
    _Spec("search", "about_filter_enabled", "SEARCH_ABOUT_FILTER_ENABLED", _opt_bool(search_constants.SEARCH_ABOUT_FILTER_ENABLED_DEFAULT)),
    _Spec("search", "llm_verify_enabled", "SEARCH_LLM_VERIFY_ENABLED", _opt_bool(search_constants.SEARCH_LLM_VERIFY_ENABLED_DEFAULT)),
    _Spec("search", "os_bm25_operator", "SEARCH_OS_BM25_OPERATOR", lambda _k: _resolve_os_bm25_operator()),
    _Spec("search", "evidence_rescue_enabled", "SEARCH_EVIDENCE_RESCUE_ENABLED", _opt_bool(search_constants.SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT)),
    _Spec("search", "evidence_debug", "SEARCH_EVIDENCE_DEBUG", _opt_bool(search_constants.SEARCH_EVIDENCE_DEBUG_DEFAULT)),
    # ── opensearch(인프라·색인 빌더 교정) ──
    _Spec("opensearch", "url", "OPENSEARCH_URL", _opt_str("http://localhost:9200")),
    _Spec("opensearch", "index", "OPENSEARCH_INDEX", _opt_str("assets")),
    _Spec("opensearch", "sync_enabled", "OPENSEARCH_SYNC_ENABLED", _opt_bool(True)),
    _Spec("opensearch", "nori_user_words", "OPENSEARCH_NORI_USER_WORDS", lambda _k: resolve_opensearch_nori_user_words()),
    _Spec("opensearch", "filename_noise_patterns", "OPENSEARCH_FILENAME_NOISE_PATTERNS", lambda _k: resolve_opensearch_filename_noise_patterns()),
    # ── relations ──
    _Spec("relations", "top_k", "RELATION_TOP_K", _opt_int(10)),
    _Spec("relations", "min_sim", "RELATION_MIN_SIM", _opt_float(0.2)),
    # 자동승인 기본 OFF(1.01 = 신뢰도가 1을 넘을 수 없어 사실상 끔 · 2026-07-31 구조 전환).
    # active("확인됨")는 **사람만** 만든다 — 근거: 자동승인분 이름표 정확도 67% 실측 +
    # conf 0.95 오분류가 강칸에 노출된 사고(same_series). 재개 조건은 specs/081 에 고정:
    # 사람이 dup 0.9 구간 150건 이상 검토 후 승인율 ≥95% 면 dup 0.9 한정 재검토.
    _Spec("relations", "auto_approve_min", "RELATION_AUTO_APPROVE_MIN", _opt_float(1.01)),
    _Spec("relations", "auto_approve_emb_min", "RELATION_AUTO_APPROVE_EMB_MIN", _opt_float(0.0)),
    # 081 승인·노출 게이트. 기본값이 새 동작을 켜지만 **전부 env 로 끌 수 있다** —
    # 관계 재생성이 전량 약 28시간이라 코드 revert 로는 즉시 되돌아오지 않는다.
    # 종류 목록은 쉼표 구분 원시 문자열로 보관하고 집합 변환은 소비처의
    # `src/relations/approval_policy.py:parse_kind_set` 이 한다(설정이 관계 어휘를 몰라도 되게).
    # 유사도 계열(duplicate_near·same_domain) 저신뢰 제안을 **행으로 만들지 않는** 하한. 0=끔.
    # 0.70 = P2 게이트(2026-07-31): 점수 어휘가 {0.9,0.7,0.5,0.3} 4값으로 바뀌어(v3 채택)
    # 0.5 이하( dup 41~46%·sd "대분야만" )를 적재에서 끊는 값. 옛 0.75 는 옛 점수 분포 기준.
    _Spec("relations", "persist_min_conf_similarity",
          "RELATION_PERSIST_MIN_CONF_SIMILARITY", _opt_float(0.70)),
    # 신뢰도와 무관하게 자동승인에서 제외할 종류. ""=제외 없음(기존 동작).
    _Spec("relations", "auto_approve_exclude_kinds",
          "RELATION_AUTO_APPROVE_EXCLUDE_KINDS", _opt_str_allow_empty("same_domain")),
    # 사람 검토 큐에서 뺄 종류(삭제 아님·필터). ""=전건 검토(기존 동작).
    _Spec("relations", "review_exempt_kinds",
          "RELATION_REVIEW_EXEMPT_KINDS", _opt_str_allow_empty("same_domain")),
    _Spec("relations", "path_top_k", "RELATION_PATH_TOP_K", _opt_int(10)),
    _Spec("relations", "retry_max_attempts", "RELATION_RETRY_MAX_ATTEMPTS", _opt_int(3)),
    # ── video(키프레임·near-dup 7필드) ──
    _Spec("video", "max_keyframes", "VIDEO_MAX_KEYFRAMES", _video_max_keyframes),
    _Spec("video", "dedup_enabled", "VIDEO_KEYFRAME_DEDUP_ENABLED", _opt_bool(_kf.DEFAULT_ENABLED)),
    _Spec("video", "dedup_hash_max", "VIDEO_KEYFRAME_DEDUP_HASH_MAX", _opt_int(_kf.DEFAULT_HASH_MAX)),
    _Spec("video", "dedup_ssim_min", "VIDEO_KEYFRAME_DEDUP_SSIM_MIN", _opt_float(_kf.DEFAULT_SSIM_MIN)),
    _Spec("video", "dedup_ssim_gray_lo", "VIDEO_KEYFRAME_DEDUP_SSIM_GRAY_LO", _opt_float(_kf.DEFAULT_SSIM_GRAY_LO)),
    _Spec("video", "dedup_hist_min", "VIDEO_KEYFRAME_DEDUP_HIST_MIN", _opt_float(_kf.DEFAULT_HIST_MIN)),
    _Spec("video", "dedup_compare_mode", "VIDEO_KEYFRAME_DEDUP_COMPARE_MODE", _opt_str(_kf.DEFAULT_COMPARE_MODE)),
    _Spec("video", "dedup_recent_window", "VIDEO_KEYFRAME_DEDUP_RECENT_WINDOW", _opt_int(_kf.DEFAULT_RECENT_WINDOW)),
    _Spec("video", "labels_meta_top_k", "VIDEO_KEYFRAME_LABELS_META_TOP_K", _opt_int(5)),
    # ── vlm(요약·이미지 라벨) ──
    _Spec("vlm", "summary_prompt_v2", "VLM_SUMMARY_PROMPT_V2", _opt_bool(False)),
    _Spec("vlm", "summary_ab_judge", "VLM_SUMMARY_AB_JUDGE", _opt_bool(False)),
    _Spec("vlm", "image_labels_meta_top_k", "IMAGE_LABELS_META_TOP_K", _opt_int(10)),
    _Spec("vlm", "labels_score_min", "LABELS_SCORE_MIN", _opt_float(0.1)),
    # ── topic ──
    _Spec("topic", "canonicalize_enabled", "TOPIC_CANONICALIZE_ENABLED", _opt_bool(False)),
)


def _build_settings(profile: Literal["dev", "prod"]) -> PipelineSettings:
    """환경변수를 읽어 설정 객체를 조립한다.

    필드 표(``_FIELD_SPECS``)를 그룹별로 모아 하위 설정부터 만들고 마지막에 전체를 조립한다 —
    필드가 표 한 줄로 선언되므로 env 키·기본값·검증이 한곳에 모인다.
    ``profile`` 만 환경변수가 아니라 인자로 받는다(어느 환경으로 띄울지는 호출자가 정한다).

    Args:
        profile: ``dev`` 또는 ``prod``.

    Returns:
        조립된 설정 객체.

    Raises:
        ValueError: 필수 환경변수 누락·형식 오류·설정 간 모순.
    """
    by_group: dict[str, dict[str, Any]] = {}
    for spec in _FIELD_SPECS:
        by_group.setdefault(spec.group, {})[spec.attr] = spec.read(spec.env)
    common = by_group.pop("", {})
    groups = {name: cls(**by_group[name]) for name, cls in _GROUP_CLASSES.items()}
    settings = PipelineSettings(profile=profile, **common, **groups)
    # 038: 단일 필드 fail-fast(_resolve_*)로 못 잡는 교차필드 불변식(OS read ⇒ OS write 필수 등)을
    # 빌드 완료 후 검증한다 — 오설정이 런타임까지 숨지 않게(init_settings=_build_settings 검증 지점).
    _validate_settings_consistency(settings)
    return settings


def init_settings(profile: Literal["dev", "prod"]) -> PipelineSettings:
    """설정을 만들어 **프로세스 전역에 고정**한다 — 실행 진입점이 가장 먼저 부른다.

    이후 어디서든 ``get_current_settings()`` 가 같은 객체를 돌려준다. 한 프로세스가 도중에 다른
    설정으로 바뀌면 앞뒤 동작이 달라지므로, 설정 확정은 기동 시 한 번뿐이어야 한다.

    Args:
        profile: ``dev`` 또는 ``prod``.

    Returns:
        확정된 설정 객체.
    """
    global _SETTINGS
    _SETTINGS = _build_settings(profile)
    return _SETTINGS


def get_current_settings() -> PipelineSettings:
    """전역에 고정된 설정을 돌려준다.

    Returns:
        ``init_settings`` 로 확정된 설정 객체.

    Raises:
        RuntimeError: 아직 초기화되지 않았을 때. **이 예외를 잡아 기본값으로 폴백하는
            코드가 여럿 있다**(설정 없이 도는 단위 테스트 경로) — 운영 진입점은 항상
            먼저 초기화하므로 그 폴백이 운영에서 쓰이면 초기화 누락이다.
    """
    if _SETTINGS is None:
        raise RuntimeError("settings가 초기화되지 않았습니다. 먼저 init_settings(profile)를 호출하세요.")
    return _SETTINGS


# E6: 지원 텍스트 임베딩 채널 화이트리스트 — 아래 model_for_channel 매핑 키의 단일 출처.
#   _validate_settings_consistency 가 기동 시점 검증에 쓰고, model_for_channel 매핑이 이 집합과 일치해야 한다.
_TEXT_EMBED_CHANNELS = frozenset({"st", "st_bge", "st_api"})


def model_for_channel(channel: str, settings: PipelineSettings | None = None) -> str:
    """텍스트 임베딩 채널 → 질의 임베딩 모델 매핑(017 A/B). 질의-문서 모델을 일치(FR-004)시키는 단일 출처.

    **적재와 질의가 같은 모델을 쓰게 하는 단일 출처**다 — 두 쪽이 각자 모델을 고르면 벡터가
    다른 공간에 놓여 검색이 조용히 엉뚱한 결과를 낸다.

    Args:
        channel: 텍스트 임베딩 채널. 시각 채널은 이 매핑 대상이 아니다.
        settings: 설정. ``None`` 이면 활성 설정을 쓴다(테스트는 주입해 순수 단위로 검증).

    Returns:
        그 채널이 쓸 모델 이름.

    Raises:
        ValueError: 모르는 채널일 때. **기본 모델로 접지 않는다** — 잘못된 모델로 검색하면
            결과가 조용히 무의미해진다.
    """
    cfg = settings if settings is not None else get_current_settings()
    mapping = {
        "st": cfg.embed.model,
        "st_bge": cfg.embed.model_bge,
        # 062: API 서빙 bge-m3. 채널→모델(서버가 아는 모델명). 백엔드(local/api)는 backend_for_channel 담당.
        "st_api": cfg.embed.api_model,
    }
    try:
        return mapping[channel]
    except KeyError:
        raise ValueError(
            f"지원하지 않는 텍스트 임베딩 채널: {channel!r} (지원: {sorted(mapping)})"
        ) from None


# 062: API 계산 백엔드를 쓰는 채널(직교 축). 그 외 채널은 로컬 SentenceTransformer.
# 채널 문자열 "st_api" 의 정본은 여기(_API_EMBED_CHANNELS)다 — backend_for_channel 이 직접 참조.
# embedding_constants 에 있던 동값 죽은 상수(EMBEDDING_KIND_ST_API)는 069 US-F 에서 제거됐다.
_API_EMBED_CHANNELS = frozenset({"st_api"})


def backend_for_channel(channel: str, settings: PipelineSettings | None = None) -> str:
    """텍스트 임베딩 채널 → 계산 백엔드 ``'local'`` | ``'api'`` (062).

    **채널과 백엔드는 다른 축**이다 — 채널은 "어느 모델(=어느 벡터 공간)", 백엔드는 "어디서
    계산하나(내 프로세스냐 원격 서버냐)"를 뜻한다.

    Args:
        channel: 텍스트 임베딩 채널.
        settings: ⚠️ **현재는 쓰이지 않는다** — 형제 함수들과 시그니처를 맞추고, 나중에 설정에
            따라 갈릴 여지를 남겨 둔 인자다.

    Returns:
        ``'local'`` 또는 ``'api'``.
    """
    return "api" if channel in _API_EMBED_CHANNELS else "local"


def active_embed_channel(settings: PipelineSettings | None = None) -> str:
    """운영 텍스트 임베딩 활성 채널(018). 적재·검색·관계가 공유하는 단일 출처.

    Args:
        settings: 설정. ``None`` 이면 활성 설정을 쓴다(테스트는 주입해 순수 단위로 검증).

    Returns:
        활성 채널 이름. 적재·검색·관계가 **모두 이 값을 봐야** 같은 벡터 공간에서 만난다.
    """
    cfg = settings if settings is not None else get_current_settings()
    return cfg.embed.active_channel


def active_embed_model(settings: PipelineSettings | None = None) -> str:
    """활성 채널의 임베딩 모델(018). ``active_embed_channel`` → ``model_for_channel`` 합성.

    Args:
        settings: 설정. ``None`` 이면 활성 설정을 쓴다.

    Returns:
        활성 채널의 모델 이름.

    Raises:
        ValueError: 활성 채널이 지원 목록 밖일 때(설정 실수를 조용히 넘기지 않는다).
    """
    return model_for_channel(active_embed_channel(settings), settings)
