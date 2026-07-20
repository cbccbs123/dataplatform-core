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
    # 062: API 텍스트 임베딩 백엔드(온프레미스 bge-m3 서빙·채널 'st_api'). 기본 로컬(active='st')이라
    # 미설정이어도 무영향 — 'st_api' 활성화 시에만 참조된다. 전부 선택 필드(_env_*_default).
    embed_api_base_url: str
    embed_api_model: str
    embed_api_key: str
    embed_api_timeout_s: float
    embed_api_batch_size: int
    embed_api_max_retries: int
    # 063: image/video CLIP 시각 임베딩(channel='clip') 생성 토글. 기본 True=기존 동작 불변(회귀 0).
    # off면 스킬이 clip EmbeddingItem 만 스킵(ST 캡션·CLIP 라벨·검색·관계 불변). 신규 셋업 opt-in.
    embed_enable_clip: bool
    text_embedding_chunk_size: int
    # 069 D8: 임베딩 청크 overlap(env TEXT_EMBED_CHUNK_OVERLAP·선택·기본 0). 요약용 overlap_size 와 별개
    # (이쪽은 임베딩 청킹 전용). 0=인접 청크 미겹침(현행·동작 불변), >0=경계 문맥 보존 opt-in.
    text_embedding_chunk_overlap: int
    text_embedding_normalize: bool
    video_max_keyframes: int
    image_labels_meta_top_k: int
    video_keyframe_labels_meta_top_k: int
    labels_score_min: float
    relation_top_k: int
    # 아래 두 필드는 관계 제안 품질 게이트(032/033)에서 추가된 것이다.
    relation_min_sim: float       # 후보 코사인 유사도 하한 — 이 미만은 LLM 에 넣지 않음 (기본 0.2)
    relation_auto_approve_min: float  # 이 이상이면 HITL 없이 자동 승인 (기본 0.9)
    # 033 FR-002: 자동승인 emb_score(후보 코사인 유사도) 하한 — 기본 0.0=무력(현 동작).
    # AND 게이트(LLM conf AND emb_score)에서 0 이면 emb 변이가 빠져 conf 단독 결정으로 회귀.
    relation_auto_approve_emb_min: float
    # 경로 신호(path_signal) 후보의 별도 한도(008, C-2). 임베딩 top_k 와 독립적으로 LIMIT 해
    # 동일 폴더 폭주를 차단한다. union 총 후보 ≤ relation_top_k + relation_path_top_k.
    relation_path_top_k: int
    # 관계 재시도/미해소 큐(relation_resolution)의 재시도 상한(009). attempts 가 이 값에 도달하면
    # decide_resolution_status 가 failed(DLQ)로 승격한다. run_relations --retry 가 소비 (기본 3).
    relation_retry_max_attempts: int
    # 020: OpenSearch 동기화(검색 엔진 도입·CQRS). url/index 는 증분 훅·복구 도구가 참조하는 선택 필드.
    # 038: opensearch_sync_enabled 기본 True — 037 로 검색이 OS 단일이 된 뒤 적재 시 증분 색인이 꺼져
    # 있으면 신규 자산이 검색에서 누락된다(PG 폴백 없음). 정합 가드(_validate_settings_consistency)가
    # backend=opensearch ∧ ¬sync 조합을 _build_settings 시점에 차단하므로 기본값·가드는 한 쌍이다.
    # (off 로 두면 run_ingest 증분 훅이 즉시 반환·opensearch-py 미import 하던 020 동작이지만, OS 백엔드
    #  하에선 가드가 false 를 불허한다 — OS-less 운영은 검색 자체가 037 전제상 불가.)
    opensearch_url: str
    opensearch_index: str
    opensearch_sync_enabled: bool
    # 021/037: 검색 read path 백엔드(CQRS·검색 엔진 도입). 037 에서 PG 검색 경로를 제거하며 'opensearch'
    # 전용으로 정리했다 — 기본값 'opensearch', 화이트리스트도 그 하나뿐이라 과거 'pg' 값은 _build_settings
    # (=init_settings)에서 즉시 ValueError 로 차단한다(fail-fast — 런타임까지 오설정이 숨지 않게,
    # 백로그 '설정 fail-late' 교정). opensearch_fusion_weights 는 클라이언트 융합(027)의 (BM25,kNN) 가중치다.
    search_backend: str
    opensearch_fusion_weights: tuple[float, float]
    # 023/027: OS 검색 버킷 게이트(robust baseline) 설정. 모두 선택 필드(021 동형) — 미설정 시
    # ``search_constants`` 단일 출처 기본값(F1). 게이트는 OS 검색 경로(037 단일 백엔드)에 적용된다.
    # eps=상대 신호(top-baseline) 하한·floor=코사인 절대 backstop. 범위 밖 값은 _resolve_* 헬퍼가 _build_settings
    # 시점에 즉시 ValueError 로 차단한다(fail-fast). 027 클라이언트 융합 전환으로 게이트 표본 수 설정·
    # 정규화 융합 검색 파이프라인 메타 필드는 제거됐다 — 게이트 신호는 같은 kNN 표본에서 직접 잰다(추가 검색 0).
    search_os_cutoff_enabled: bool
    search_os_cutoff_eps: float
    search_os_cutoff_floor: float
    # 027: OS per-result 컷 코사인 하한(단일 코사인 스케일). 024 의 모달리티별 정규화 스케일 임계 4종을
    # 대체하는 전역 1개 — 행 유지 = BM25 매칭 OR 원시 코사인 ≥ 이 값.
    # 범위 [−1,1] 밖은 _resolve_os_result_floor 가 _build_settings 시점에 즉시 ValueError(fail-fast).
    search_os_result_floor: float
    # 028: reranker 평가(쌍별 절대 판정·추론만). 기본 off — 평가 opt-in(회귀 0). τ∈[0,1]·R≥1 fail-fast.
    search_os_rerank_enabled: bool
    search_os_rerank_model: str
    search_os_rerank_top_r: int
    search_os_rerank_tau: float
    # 029: LLM 질의 명사구 정규화 토글(021 FR-004 토글 개정). 기본 off — 미설정 시 검색시점 LLM 미실행
    # (027 바이트 동일·회귀 0). on 이면 검색 직전 질의를 gemma 명사구로 정규화(temp=0·env 입력 0·단일
    # seam)해 임베딩·BM25 양쪽에 동일 적용한다. 순수 토글이라 범위검증 불필요(_env_bool_default·cutoff 동형).
    search_os_query_norm_enabled: bool
    # 075: 질의 정규화 방식. "morph"(nori 형태소·072·기본) | "llm"(gemma·029). enabled on 일 때만 유효.
    search_os_query_norm_method: str
    # 073: aboutness OR-증거 필터 토글. 기본 off(회귀 0). on 이면 적재시 확정한 about 개체+keywords 를
    # 증거로 질의 개체와 무증거 행을 버킷에서 걸러낸다(검색시점 LLM 0·전체 노출 깊이). 백필 후 opt-in.
    # ⚠️ 운영 전제(리뷰 지적): 필터는 질의가 **명사구**라고 가정한다(query.split()=명사 리스트) —
    # 072 query-norm(search_os_query_norm_enabled) on 과 함께 켜는 것을 전제로 측정·채택됐다.
    # norm off + filter on 이면 조사 포함 어절로 kmatch 실효가 떨어진다(fail-safe 로 안전하나 비권장).
    search_about_filter_enabled: bool
    # 074: 검색시점 top-3 개별 LLM 검증(L2) 토글. 기본 off(회귀 0). on 이면 자연어(어절≥3) 질의의
    # 상위 3 자산을 gemma 개별 병렬 판정해 무관 제거(데드라인 1.5s 전량 폴백·판정 캐시). 029 선례.
    search_llm_verify_enabled: bool
    # 025: OS BM25 multi_match operator. 기본 'or'(현행 본문 불변·회귀 0), 'and'=질의 전 토큰 매칭
    # 요구(복합어 부분토큰 가짜매칭 F2 차단 — 의미 매칭은 kNN 보완). 화이트리스트 밖은 즉시 ValueError.
    search_os_bm25_operator: str
    # 044: 필드 evidence 기반 lexical rescue — ``search_constants`` 와 쌍. 임계·가중·seed 는 상수
    # 모듈 단일 출처; 여기는 **런타임 on/off·관측** 만.
    search_evidence_rescue_enabled: bool
    # env ``SEARCH_EVIDENCE_RESCUE_ENABLED``. 게이트 실패 버킷에서 BM25 구제 행을
    # ``lexical_rescue_keep``(policy·strong/weak·임계)로 걸러낼지. False=027 legacy(어휘 hit 전부
    # keep). True=044 live — 예: q=테스트 auto·restricted 시 summary weak-only(낚시) drop,
    # keywords strong(반도체) keep. cosine 게이트 통과 행은 불변(invariant 1).
    search_evidence_debug: bool
    # env ``SEARCH_EVIDENCE_DEBUG``. True 시 검색 결과 각 hit 에 ``matched_queries``·
    # ``evidence_score``·``strong_evidence_score``·``gate_passed``·``keep_reason`` 부착(run_search·
    # search_hybrid 경로). 스모크·골든 원인 분석용 — 운영 기본 off.
    # 045 v2a: ``SEARCH_GENERIC_TERM_SEED_EXTRA`` + core seed merge(NFKC dedup) — ``query_plan`` resolver.
    search_generic_term_seed: tuple[str, ...]
    # 026: OS 색인 빌더 교정용 선택 설정(021/023 동형 — 미설정 시 기존 동작 불변). 모두 OS 색인 한정
    # (pg FTS 무접촉). opensearch_nori_user_words = 커스텀 nori analyzer 의 user_dictionary 외래어 목록
    # (build_index_body 기본과 동치). opensearch_filename_noise_patterns = 파일명 정제 추가 regex(기본 빈).
    opensearch_nori_user_words: tuple[str, ...]
    opensearch_filename_noise_patterns: tuple[str, ...]
    # 048: 영상 키프레임 near-dup 제거(VLM 전 결정적 2단계 dHash→SSIM/HSV). 임계·모드 단일 출처(FR-501).
    # 모두 _env_*_default 선택 필드 — 미설정 시 spec 기본설정표 값. video_skill 배선(G3)·dedup 코어가
    # 이 7필드로 KeyframeDedupConfig 를 만든다(하드코딩 분산 금지). enabled 기본 True(사용자 결정
    # 2026-06-29) 이나 off 경로(FR-103)는 언제든 추출 바이트 동일이라 회귀 안전판이다.
    video_keyframe_dedup_enabled: bool
    video_keyframe_dedup_hash_max: int
    video_keyframe_dedup_ssim_min: float
    video_keyframe_dedup_ssim_gray_lo: float
    video_keyframe_dedup_hist_min: float
    video_keyframe_dedup_compare_mode: str
    video_keyframe_dedup_recent_window: int
    # 049: VLM 요약 프롬프트 v2 토글(FR-101·FR-601). 캡션(image_summarizer)·reduce(video_summarizer)
    # 의 v1/v2 프롬프트와 키워드 후처리(objects 승격)를 고르는 단일 출처. 모두 _env_bool_default 선택
    # 필드 — 미설정 시 기본 False. vlm_summary_prompt_v2=False(기본) 면 summarize_* 가 빌더에 v2=False
    # 를 넘기고 v1 inline 키워드 루프를 그대로 써, 추출 결과가 현행과 **바이트 동일**하다(FR-102 회귀
    # 안전판). vlm_summary_ab_judge 는 A/B 측정 하니스(G4)의 LLM-judge 옵션 — 평가용·추출 무영향.
    vlm_summary_prompt_v2: bool
    vlm_summary_ab_judge: bool
    # 058: 관계 topic 정본화 배선 토글(FR-401). graph_persist 가 persist 직전 topic/subtopic 을
    # canonicalize_topic/canonicalize_subtopic 로 정본화할지 고르는 단일 출처. _env_bool_default
    # 선택 필드(029/049 동형) — 미설정 시 기본 **False**. 빈 레지스트리에서 켜면 raw topic 이 전부
    # 자동등록(부작용)돼 시드 전 동작이 깨지므로 기본 off(동작 불변·시드 전 동치); 시드(G5) 후 명시적
    # 활성화한다. False 면 sync_graph_edges 가 coerce_topic_fields_mvp 결과를 그대로 저장한다(회귀 0).
    topic_canonicalize_enabled: bool


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
    if settings.search_backend == "opensearch" and not settings.opensearch_sync_enabled:
        raise ValueError(
            "설정 불일치: SEARCH_BACKEND=opensearch 인데 OPENSEARCH_SYNC_ENABLED=false 입니다. "
            "OS 검색은 적재 시 OS 증분 색인이 필수입니다(037 이후 PG 폴백 없음 — 끄면 신규 자산이 "
            "검색에서 누락). OPENSEARCH_SYNC_ENABLED=true 로 설정하세요."
        )
    # 062: API 임베딩 채널(st_api) 활성인데 base_url 미설정이면 파이프라인 한복판(/embeddings 호출)이
    #   아니라 기동 시점에 즉시 차단한다(038 fail-fast 관례와 통일 — 채널만 켜는 사람 실수 방지).
    if backend_for_channel(settings.active_embed_channel, settings) == "api" and not settings.embed_api_base_url:
        raise ValueError(
            "설정 불일치: EMBED_ACTIVE_CHANNEL=st_api(API 임베딩) 인데 EMBED_API_BASE_URL 이 비어 있습니다. "
            "API 백엔드는 엔드포인트 주입이 필수입니다 — EMBED_API_BASE_URL 을 설정하세요(예: http://<host>:<port>/v1)."
        )
    # 069 D8: 임베딩 청크 overlap 은 iter_document_chunks 가 0<=overlap<chunk_size 를 요구한다. 위반 시
    #   첫 문서 처리 시점(파이프라인 한복판)에야 ValueError 가 터지므로, opt-in 오설정을 기동 시점에
    #   즉시 차단한다(038/062 fail-fast 관례와 통일). 기본 0 은 항상 통과(동작 불변).
    if not (0 <= settings.text_embedding_chunk_overlap < settings.text_embedding_chunk_size):
        raise ValueError(
            f"설정 불일치: TEXT_EMBED_CHUNK_OVERLAP={settings.text_embedding_chunk_overlap} 는 "
            f"0 이상이고 TEXT_EMBED_CHUNK_SIZE={settings.text_embedding_chunk_size} 미만이어야 합니다 "
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


def resolve_search_generic_term_seed_extra() -> tuple[str, ...]:
    """045 v2a — ``SEARCH_GENERIC_TERM_SEED_EXTRA`` CSV. 미설정 시 빈. 공백 항목은 fail-fast."""
    raw = os.getenv(search_constants.SEARCH_GENERIC_TERM_SEED_EXTRA_ENV)
    if raw is None or not raw.strip():
        return ()
    parts = [p.strip() for p in raw.split(",")]
    if any(not p for p in parts):
        raise ValueError(
            f"generic seed 빈 항목: {search_constants.SEARCH_GENERIC_TERM_SEED_EXTRA_ENV}={raw!r}"
        )
    return tuple(parts)


def resolve_search_generic_term_seed() -> tuple[str, ...]:
    """core ``GENERIC_SINGLE_TERM_SEED`` + env extra merge(결정적 dedup)."""
    from src.search.query_plan import merge_generic_term_seed

    return merge_generic_term_seed(
        search_constants.GENERIC_SINGLE_TERM_SEED,
        resolve_search_generic_term_seed_extra(),
    )


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
    return lambda key: _env_int_default(key, default)


def _opt_str(default: str) -> Callable[[str], str]:
    return lambda key: _env_str_default(key, default)


def _opt_float(default: float) -> Callable[[str], float]:
    return lambda key: _env_float_default(key, default)


def _opt_bool(default: bool) -> Callable[[str], bool]:
    return lambda key: _env_bool_default(key, default)


def _video_max_keyframes(key: str) -> int:
    # 0/음수 입력은 48 로 보정(현행 동작 보존) — 특이 로직이라 전용 read.
    vk = _env_int_default(key, 48)
    return 48 if vk <= 0 else vk


class _Spec(NamedTuple):
    """한 필드 = 한 행: 속성명·env 키·읽기함수·필수여부."""

    attr: str
    env: str
    read: Callable[[str], Any]
    required: bool = False


# ── 069 US-E FR-E4(PR4a): 필드 명세 단일 출처 ──────────────────────────────────
# 종전 _build_settings 는 64필드를 손으로 나열했고 테스트는 격리할 선택 env 키를 또 따로 나열했다
# (두 곳 드리프트 위험). 아래 _FIELD_SPECS 한 테이블에서 build 조립·테스트 격리키·커버리지가 모두
# 파생된다 — 새 필드 추가 시 이 테이블 한 행만 늘리면 된다(SC-E). 위 dataclass 필드 정의와 이 테이블은
# test_settings.TestFieldSpecsSSOT 가 1:1 로 봉인한다.
#
# read 는 env 키 하나를 받아 값을 돌려주는 함수다(기본값·범위검증 캡슐화). 세 종류:
#   - 필수(_require_env*)       : 미설정 시 ValueError. required=True.
#   - 선택(_opt_*(기본값))       : 미설정 시 기본값. required=False.
#   - 검증 resolver(_resolve_*) : 화이트리스트·범위 fail-fast 를 자체 수행(env 키가 내부 하드코드라
#                                 read 는 인자를 무시하고 호출만; spec 의 env 는 격리키 파생용 메타).
_FIELD_SPECS: tuple[_Spec, ...] = (
    # 필수(_require_env*) — 미설정 시 ValueError.
    _Spec("meta_model", "META_MODEL", _require_env, required=True),
    _Spec("openai_base_url", "OPENAI_BASE_URL", _require_env, required=True),
    _Spec("openai_api_key", "OPENAI_API_KEY", _require_env, required=True),
    _Spec("summary_max_chars", "SUMMARY_MAX_CHARS", _require_env_int, required=True),
    _Spec("top_k_keywords", "TOP_K_KEYWORDS", _require_env_int, required=True),
    _Spec("chunk_size", "CHUNK_SIZE", _require_env_int, required=True),
    _Spec("overlap_size", "OVERLAP_SIZE", _require_env_int, required=True),
    _Spec("encoding", "ENCODING", _require_env, required=True),
    _Spec("text_embedding_model", "TEXT_EMBED_MODEL", _require_env, required=True),
    _Spec("text_embedding_chunk_size", "TEXT_EMBED_CHUNK_SIZE", _require_env_int, required=True),
    _Spec("text_embedding_normalize", "TEXT_EMBED_NORMALIZE", _require_env_bool, required=True),
    # 017/018/062: 임베딩 채널·API 백엔드(기본 로컬 'st' — 미설정 무영향).
    _Spec("text_embedding_model_bge", "TEXT_EMBED_MODEL_BGE", _opt_str("BAAI/bge-m3")),
    _Spec("active_embed_channel", "EMBED_ACTIVE_CHANNEL", _opt_str("st")),
    _Spec("embed_api_base_url", "EMBED_API_BASE_URL", _opt_str("")),
    _Spec("embed_api_model", "EMBED_API_MODEL", _opt_str("BAAI/bge-m3")),
    _Spec("embed_api_key", "EMBED_API_KEY", _opt_str("")),
    _Spec("embed_api_timeout_s", "EMBED_API_TIMEOUT_S", _opt_float(30.0)),
    _Spec("embed_api_batch_size", "EMBED_API_BATCH_SIZE", _opt_int(32)),
    _Spec("embed_api_max_retries", "EMBED_API_MAX_RETRIES", _opt_int(2)),
    # 063: clip 임베딩 토글(기본 True=회귀 0).
    _Spec("embed_enable_clip", "EMBED_ENABLE_CLIP", _opt_bool(True)),
    # 069 D8: 임베딩 청크 overlap(기본 0=현행). chunk_size 와의 관계는 _validate 가 fail-fast.
    _Spec("text_embedding_chunk_overlap", "TEXT_EMBED_CHUNK_OVERLAP", _opt_int(0)),
    # video_max_keyframes: 0/음수 보정 특이 로직(전용 read).
    _Spec("video_max_keyframes", "VIDEO_MAX_KEYFRAMES", _video_max_keyframes),
    _Spec("image_labels_meta_top_k", "IMAGE_LABELS_META_TOP_K", _opt_int(10)),
    _Spec("video_keyframe_labels_meta_top_k", "VIDEO_KEYFRAME_LABELS_META_TOP_K", _opt_int(5)),
    _Spec("labels_score_min", "LABELS_SCORE_MIN", _opt_float(0.1)),
    # 관계 제안 게이트(032/033/008/009).
    _Spec("relation_top_k", "RELATION_TOP_K", _opt_int(10)),
    _Spec("relation_min_sim", "RELATION_MIN_SIM", _opt_float(0.2)),
    _Spec("relation_auto_approve_min", "RELATION_AUTO_APPROVE_MIN", _opt_float(0.9)),
    _Spec("relation_auto_approve_emb_min", "RELATION_AUTO_APPROVE_EMB_MIN", _opt_float(0.0)),
    _Spec("relation_path_top_k", "RELATION_PATH_TOP_K", _opt_int(10)),
    _Spec("relation_retry_max_attempts", "RELATION_RETRY_MAX_ATTEMPTS", _opt_int(3)),
    # 020/038: OpenSearch 동기화(기본 sync=True — backend=opensearch ∧ ¬sync 는 _validate 가 차단).
    _Spec("opensearch_url", "OPENSEARCH_URL", _opt_str("http://localhost:9200")),
    _Spec("opensearch_index", "OPENSEARCH_INDEX", _opt_str("assets")),
    _Spec("opensearch_sync_enabled", "OPENSEARCH_SYNC_ENABLED", _opt_bool(True)),
    # 021/037/027: 검색 백엔드·융합·게이트(검증 resolver — 화이트리스트·범위 fail-fast).
    _Spec("search_backend", "SEARCH_BACKEND", lambda _k: _resolve_search_backend()),
    _Spec("opensearch_fusion_weights", "OPENSEARCH_FUSION_WEIGHTS", lambda _k: _resolve_opensearch_fusion_weights()),
    _Spec("search_os_cutoff_enabled", "SEARCH_OS_CUTOFF_ENABLED", lambda _k: _resolve_os_cutoff_enabled()),
    _Spec("search_os_cutoff_eps", "SEARCH_OS_CUTOFF_EPS", lambda _k: _resolve_os_cutoff_eps()),
    _Spec("search_os_cutoff_floor", "SEARCH_OS_CUTOFF_FLOOR", lambda _k: _resolve_os_cutoff_floor()),
    _Spec("search_os_result_floor", "SEARCH_OS_RESULT_FLOOR", lambda _k: _resolve_os_result_floor()),
    # 028: rerank 평가(기본 off).
    _Spec("search_os_rerank_enabled", "SEARCH_OS_RERANK_ENABLED", _opt_bool(search_constants.OS_RERANK_ENABLED_DEFAULT)),
    _Spec("search_os_rerank_model", "SEARCH_OS_RERANK_MODEL", _opt_str(search_constants.OS_RERANK_MODEL_DEFAULT)),
    _Spec("search_os_rerank_top_r", "SEARCH_OS_RERANK_TOP_R", lambda _k: _resolve_os_rerank_top_r()),
    _Spec("search_os_rerank_tau", "SEARCH_OS_RERANK_TAU", lambda _k: _resolve_os_rerank_tau()),
    # 029/075: 질의 정규화 토글·방식.
    _Spec("search_os_query_norm_enabled", "SEARCH_OS_QUERY_NORM_ENABLED", _opt_bool(search_constants.OS_QUERY_NORM_ENABLED_DEFAULT)),
    _Spec("search_os_query_norm_method", "SEARCH_OS_QUERY_NORM_METHOD", lambda _k: _resolve_os_query_norm_method()),
    # 073/074: aboutness 필터·LLM 검증(기본 off).
    _Spec("search_about_filter_enabled", "SEARCH_ABOUT_FILTER_ENABLED", _opt_bool(search_constants.SEARCH_ABOUT_FILTER_ENABLED_DEFAULT)),
    _Spec("search_llm_verify_enabled", "SEARCH_LLM_VERIFY_ENABLED", _opt_bool(search_constants.SEARCH_LLM_VERIFY_ENABLED_DEFAULT)),
    # 025: BM25 operator(기본 or·화이트리스트 fail-fast).
    _Spec("search_os_bm25_operator", "SEARCH_OS_BM25_OPERATOR", lambda _k: _resolve_os_bm25_operator()),
    # 044: evidence rescue/debug.
    _Spec("search_evidence_rescue_enabled", "SEARCH_EVIDENCE_RESCUE_ENABLED", _opt_bool(search_constants.SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT)),
    _Spec("search_evidence_debug", "SEARCH_EVIDENCE_DEBUG", _opt_bool(search_constants.SEARCH_EVIDENCE_DEBUG_DEFAULT)),
    # 045: generic term seed(core + env extra merge). env 키는 격리키 파생용 메타.
    _Spec("search_generic_term_seed", search_constants.SEARCH_GENERIC_TERM_SEED_EXTRA_ENV, lambda _k: resolve_search_generic_term_seed()),
    # 026: OS 색인 빌더 교정(빈 항목·컴파일 불가 fail-fast).
    _Spec("opensearch_nori_user_words", "OPENSEARCH_NORI_USER_WORDS", lambda _k: resolve_opensearch_nori_user_words()),
    _Spec("opensearch_filename_noise_patterns", "OPENSEARCH_FILENAME_NOISE_PATTERNS", lambda _k: resolve_opensearch_filename_noise_patterns()),
    # 048: 영상 키프레임 near-dup 제거 7필드(기본값 keyframe_dedup_defaults 단일 출처).
    _Spec("video_keyframe_dedup_enabled", "VIDEO_KEYFRAME_DEDUP_ENABLED", _opt_bool(_kf.DEFAULT_ENABLED)),
    _Spec("video_keyframe_dedup_hash_max", "VIDEO_KEYFRAME_DEDUP_HASH_MAX", _opt_int(_kf.DEFAULT_HASH_MAX)),
    _Spec("video_keyframe_dedup_ssim_min", "VIDEO_KEYFRAME_DEDUP_SSIM_MIN", _opt_float(_kf.DEFAULT_SSIM_MIN)),
    _Spec("video_keyframe_dedup_ssim_gray_lo", "VIDEO_KEYFRAME_DEDUP_SSIM_GRAY_LO", _opt_float(_kf.DEFAULT_SSIM_GRAY_LO)),
    _Spec("video_keyframe_dedup_hist_min", "VIDEO_KEYFRAME_DEDUP_HIST_MIN", _opt_float(_kf.DEFAULT_HIST_MIN)),
    _Spec("video_keyframe_dedup_compare_mode", "VIDEO_KEYFRAME_DEDUP_COMPARE_MODE", _opt_str(_kf.DEFAULT_COMPARE_MODE)),
    _Spec("video_keyframe_dedup_recent_window", "VIDEO_KEYFRAME_DEDUP_RECENT_WINDOW", _opt_int(_kf.DEFAULT_RECENT_WINDOW)),
    # 049: VLM 요약 프롬프트 v2 토글(기본 False=v1 바이트 동일).
    _Spec("vlm_summary_prompt_v2", "VLM_SUMMARY_PROMPT_V2", _opt_bool(False)),
    _Spec("vlm_summary_ab_judge", "VLM_SUMMARY_AB_JUDGE", _opt_bool(False)),
    # 058: 관계 topic 정본화 토글(기본 False=동작 불변).
    _Spec("topic_canonicalize_enabled", "TOPIC_CANONICALIZE_ENABLED", _opt_bool(False)),
)


def _build_settings(profile: Literal["dev", "prod"]) -> PipelineSettings:
    # 종전 64필드 손나열 조립을 _FIELD_SPECS 단일 출처 루프로 대체(동작 불변 — read 별로 종전과
    # 동일한 env 키·기본값·검증을 호출). profile 만 env 유래가 아니라 직접 주입한다.
    values = {spec.attr: spec.read(spec.env) for spec in _FIELD_SPECS}
    settings = PipelineSettings(profile=profile, **values)
    # 038: 단일 필드 fail-fast(_resolve_*)로 못 잡는 교차필드 불변식(OS read ⇒ OS write 필수 등)을
    # 빌드 완료 후 검증한다 — 오설정이 런타임까지 숨지 않게(init_settings=_build_settings 검증 지점).
    _validate_settings_consistency(settings)
    return settings


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
        # 062: API 서빙 bge-m3. 채널→모델(서버가 아는 모델명). 백엔드(local/api)는 backend_for_channel 담당.
        "st_api": cfg.embed_api_model,
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

    ``'st_api'``=온프레미스 API 서빙(bge-m3·``/v1/embeddings``), 그 외(``'st'``·``'st_bge'``)=로컬
    SentenceTransformer. 채널은 "무슨 모델(공간)", 백엔드는 "어떻게 계산" — 018 채널 위에 얹는 직교 축.
    ``settings`` 인자는 시그니처 대칭용(현재 매핑은 채널만으로 결정·향후 확장 여지)."""
    return "api" if channel in _API_EMBED_CHANNELS else "local"


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
