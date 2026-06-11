"""하이브리드 검색 서비스 진입점 — 호출부(CLI/HTTP)에 독립적인 함수 계층.

요청(query·modalities·limit·alpha)을 받아 ``search_media_all_grouped`` 로 모달리티별
버킷 결과를 만든 뒤, 요청한 모달리티만 골라 일정한 모양으로 반환한다. 실제 검색·LLM·DB는
``media_search`` 가 담당하고, 본 모듈은 요청 정규화·필터·응답 형태만 책임진다(F-4.3, 단계 C+).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from src.config.settings import (
    ChunkAggConfig,
    active_embed_channel,
    get_current_settings,
    model_for_channel,
)
from src.search.media_search import EMBEDDING_KIND_ST, search_media_all_grouped

# 021 G3: OpenSearch 백엔드 분기용 seam. opensearch_search 모듈 상단은 순수(opensearch-py·임베더는
# 함수 내부 지연 import)라 pg 기본 환경에서도 import 안전 — 실제 OS IO 는 backend='opensearch' 호출
# 시에만 발생한다(플래그 off 순수성 보존).
from src.search.opensearch_search import get_client as os_get_client
from src.search.opensearch_search import search_assets_os as os_search_assets

_LOG = logging.getLogger(__name__)

# 요청 모달리티 라벨 → ``search_media_all_grouped`` 결과 버킷 키.
# 결정성(헌법 3조): 결과 버킷 조립이 ``list(.items())`` 순회 순서에 의존하므로 삽입 순서를
# 보존한다(dict 는 3.7+ 삽입 순서 보장). set 등 순서 비보장 타입으로 대체 금지.
_MODALITY_BUCKETS: dict[str, str] = {
    "text": "text_documents",
    "audio": "audio",
    "image": "image",
    "video": "video",
}

# 022 백엔드 분담: backend='opensearch' 면 text·audio·**image·video 모두** 020 OS 인덱스(하이브리드)에서
# 검색한다(021 의 image/video→PG CLIP 경로를 OS 로 전환). image/video 는 020 assets 인덱스에 한국어 VLM
# 캡션(nori) + KoSimCSE 캡션 임베딩(embedding)으로 이미 색인돼 있어 text/audio 와 동일 하이브리드로
# 회수된다(CLIP 아님 — 시각-내용 매칭은 후속 spec). 따라서 OS 경로는 요청 모달리티 전체를 한 번의
# search_assets_os 호출로 처리하며, PG/모달리티 분기(021 의 _OS_MODALITIES·_PG_VISUAL_MODALITIES)는 제거됐다.

# OS 정규화 융합 가중치 (BM25, kNN) 기본값. 설정 필드(opensearch_fusion_weights) 정식화·범위검증은
# G4(T007) — 미설정이면 이 측정 근거 균형값으로 폴백(getattr). 동작 불변(SC-001): pg 경로 무관.
_DEFAULT_OS_FUSION_WEIGHTS: tuple[float, float] = (0.5, 0.5)

# 023: OS 검색 적합도 컷오프(probe 게이트) 기본값. _DEFAULT_OS_FUSION_WEIGHTS 동형 — settings 미초기화
# (순수 단위 등)에서 getattr 폴백으로 쓴다(cross-module private import 안 함). enabled 기본 False 는
# 미초기화 시 안전(무게이트) — 실 settings 기본은 True 이나 search_backend='pg'(기본)면 OS 경로 미실행
# 이라 무관하다. eps/floor/probe_k 는 settings 기본과 같은 측정 근거값(opensearch_search 의 G1/G2 상수 동치).
_DEFAULT_OS_CUTOFF_ENABLED: bool = False
_DEFAULT_OS_CUTOFF_EPS: float = 0.15  # G4 calibration 확정(opensearch_search 상수 동치)
_DEFAULT_OS_CUTOFF_FLOOR: float = 0.43
_DEFAULT_OS_PROBE_K: int = 50


def _row_similarity(row: dict[str, Any]) -> float:
    """행의 ``similarity`` 를 유한 실수로 읽는다(None/NaN/inf/비수치 → 0.0).

    media_search 의 비공개 헬퍼에 의존하지 않도록 서비스 계층에 작은 정화 함수를 둔다.
    """
    value = row.get("similarity")
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def _filter_by_min_score(
    rows: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """``similarity`` 가 ``threshold`` 미만인 행을 제거한다. 0.0 이하(음수 포함)면 필터 비활성(원본 반환).

    grouped 결과는 이미 ``similarity`` 내림차순 상위 N 건이라, 잘린 후보는 남은 것보다 점수가
    더 낮다 — 따라서 cap 이후 이 계층에서 걸러도 누락되는 적합 자산은 없다.
    ``NaN``/누락 ``similarity`` 는 ``_row_similarity`` 로 0.0 처리되어 임계값>0 이면 자연 탈락한다.
    """
    if not threshold or threshold <= 0.0:
        return rows
    return [r for r in rows if _row_similarity(r) >= threshold]


def _grouped_via_opensearch(
    query: str,
    *,
    modalities: list[str] | None,
    limit_per_bucket: int,
    text_channel: str,
    cfg: Any,
    os_search_fn: Callable[..., dict[str, list[dict[str, Any]]]],
    os_client_fn: Callable[..., Any],
) -> dict[str, Any]:
    """backend='opensearch' 경로의 모달리티 버킷을 조립한다(022, FR-002·FR-003·SC-005).

    text·audio·**image·video 모든 버킷**을 020 OS 인덱스에서 동일 하이브리드(nori BM25 캡션·라벨 +
    ``embedding`` kNN + 정규화 융합)로 검색한다 — image/video 도 020 assets 인덱스에 한국어 VLM 캡션·
    KoSimCSE 캡션 임베딩으로 색인돼 있어 text/audio 와 같은 경로다(CLIP 아님; 시각-내용 매칭은 후속 spec).
    요청 모달리티 전체를 **한 번의** ``os_search_fn`` 호출로 검색해 버킷을 만들고, 현 pg 분기와 같은
    키(text_documents·audio·image·video·meta)로 담는다 — 반환 grouped 는 pg 분기와 동일 모양이라 호출부의
    ``_filter_by_min_score`` 공유 코드가 그대로 처리한다(응답 동형).

    설계 판단:
    - **PG·LLM 미접촉(FR-002·SC-004)**: PG grouped(시각 CLIP 2단계)·``structure_user_query``(LLM)를
      호출하지 않는다 — 원문 ``query`` 만 OS 에 넘긴다(멀티모달 LLM 0·ms). 따라서 PG 전용 파라미터
      (structured·alpha·fusion·query_model_name·chunk_agg·grouped_fn)는 이 경로에서 제거됐다.
    - **모달리티 키 매핑**: ``os_search_fn`` 버킷 키는 모달리티명('text'/'image')이고 응답 grouped 키는
      ('text_documents'/'image')이므로 ``_MODALITY_BUCKETS`` 로 변환해 담는다.
    - **meta**: 항상 ``{"backend": "opensearch"}`` 단일(PG meta 통과 없음 — grouped 미호출).
    - **OS 미도달(FR-007)**: ``os_client_fn``/``os_search_fn`` 예외를 try/except 로 감싸지 않아 그대로
      전파한다(silent pg 폴백 금지 — 결과가 백엔드 가용성에 따라 달라지지 않게).

    ⚠️ 결정성(헌법 3조): 최종 응답 버킷 순서는 호출부의 ``label_keys`` 가 정하므로 여기 grouped 의
    삽입 순서는 출력 순서에 영향하지 않는다.
    """
    requested = modalities if modalities is not None else list(_MODALITY_BUCKETS)

    grouped: dict[str, Any] = {"meta": {"backend": "opensearch"}}
    if not requested:
        return grouped  # 빈 요청: OS 미접촉(불필요 IO 회피)

    client = os_client_fn()  # OS 클라이언트 생성 실패 시 예외 전파(FR-007)
    os_buckets = os_search_fn(
        client,
        query,
        modalities=requested,  # 요청 전 모달리티(image/video 포함)를 한 번에 OS 검색
        k=limit_per_bucket,
        channel=text_channel,
        weights=getattr(cfg, "opensearch_fusion_weights", _DEFAULT_OS_FUSION_WEIGHTS),
        index=getattr(cfg, "opensearch_index", "assets"),
        pipeline_name=getattr(cfg, "opensearch_search_pipeline", "assets-hybrid"),
        exclude_medical=True,
        # 023: 적합도 컷오프(probe 게이트) 설정을 cfg 에서 읽어 OS seam 에 전달한다(fusion_weights 동형
        # getattr 폴백 — settings 미초기화 순수 단위 방어). cutoff_enabled=False(미초기화 기본)면
        # search_assets_os 가 probe 미호출·버킷 그대로라 021/022 동작 동치다(회귀 0).
        cutoff_enabled=getattr(cfg, "search_os_cutoff_enabled", _DEFAULT_OS_CUTOFF_ENABLED),
        cutoff_eps=getattr(cfg, "search_os_cutoff_eps", _DEFAULT_OS_CUTOFF_EPS),
        cutoff_floor=getattr(cfg, "search_os_cutoff_floor", _DEFAULT_OS_CUTOFF_FLOOR),
        cutoff_probe_k=getattr(cfg, "search_os_probe_k", _DEFAULT_OS_PROBE_K),
    )  # client.search 미도달 예외도 전파(FR-007)
    # 모달리티명('text'/'image') → grouped 버킷 키('text_documents'/'image') 매핑.
    for m in requested:
        grouped[_MODALITY_BUCKETS[m]] = os_buckets.get(m, [])

    return grouped


def search_hybrid(
    query: str,
    *,
    modalities: list[str] | None = None,
    limit_per_bucket: int = 20,
    text_hybrid_alpha: float = 0.75,
    image_search_alpha: float = 0.65,
    fusion: str = "alpha",
    structured: dict[str, Any] | None = None,
    min_scores: dict[str, float] | None = None,
    text_channel: str | None = None,
    text_query_model: str | None = None,
    chunk_agg: ChunkAggConfig | None = None,
    backend: str | None = None,
    _grouped_fn: Callable[..., dict[str, Any]] = search_media_all_grouped,
    _os_search_fn: Callable[..., dict[str, list[dict[str, Any]]]] = os_search_assets,
    _os_client_fn: Callable[..., Any] = os_get_client,
) -> dict[str, Any]:
    """질의를 하이브리드 검색해 모달리티 버킷으로 반환한다.

    ``modalities`` 가 ``None`` 이면 전체 버킷(text/audio/image/video)을, 지정하면 해당
    버킷만 반환한다. 알 수 없는 모달리티 라벨은 ``ValueError``. 요청에 image·video 가
    하나도 없으면(text/audio 전용) grouped 에 ``include_visual=False`` 를 넘겨 시각 2단계
    (CLIP)를 건너뛴다 — 버려질 시각 후보를 계산하지 않아 비용↓, 반환 버킷은 동일.
    ``structured`` 를 넘기면 그대로 grouped 검색에 전달돼 LLM 질의 구조화를 건너뛴다
    (이미 구조화됐거나 LLM 없이 테스트할 때). ``_grouped_fn`` 은 테스트 주입 seam.
    ``min_scores`` 는 모달리티 라벨→적합도 하한(0.0=비활성); 각 버킷에서 ``similarity`` 가
    임계값 미만인 행을 응답에서 제외한다(미지정 모달리티는 필터하지 않음).
    ``fusion`` 은 ST 하이브리드 융합 방식(기본 ``alpha``=기존 동작; ``rrf``=순위 융합 프로토타입).
    ⚠️ 한계: ``rrf`` 는 현재 grouped 출력에 반영되지 않는다 — 버킷 cap 이 ``similarity`` 로
    재정렬하므로 RRF 순서는 ``_run_hybrid_search`` 레벨에서만 효과(KPI 측정용). 설계 §8 후속.
    ``chunk_agg`` 는 per-asset 청크 집계 방식(019)이다. 미지정(None)이면 검색 SQL 빌더 호출부가
    ``chunk_agg_config()`` 로 활성 설정을 해소한다(기본 ``max``=종전 동치, 회귀 0). 명시 전달은
    그대로 grouped 경로로 흘러 우선한다(017 채널처럼 측정 seam — KPI 하니스가 집계 방식을 주입).

    ``text_channel``/``text_query_model`` 은 텍스트 임베딩 채널 선택이다(텍스트 채널 한정, 시각
    CLIP 경로 무변경). **미지정(None)** 이면 운영 활성 프로파일(018, 적재·검색·관계 단일 출처)로
    해소한다 — ``text_channel`` 은 ``active_embed_channel()``, 질의 모델은 (해소된) 채널의
    ``model_for_channel`` 로 일치시킨다(FR-004 질의-문서 모델 일치). 017 A/B 하니스처럼 **명시
    전달은 그대로 우선**한다(명시 채널/모델이면 활성 해소를 건너뜀).

    회귀 0(SC-002): 기본 active='st' → channel='st'·KoSimCSE(=``model_for_channel('st')``=
    ``cfg.text_embedding_model``) 로 기존 동작과 동치. settings 미초기화(순수 단위 등)에서는
    활성 해소가 ``RuntimeError`` 이므로 기존 기본 ``('st', None)`` 으로 보수적 폴백한다 —
    이때 ``query_model_name=None`` 을 넘겨 media_search 가 기존대로 KoSimCSE 로 해소한다(006/017
    검색 단위가 settings 없이 그대로 동작). 미지원 채널은 ``model_for_channel`` 이 ``ValueError``.

    ``backend`` 는 검색 read path 백엔드(021)다. **미지정(None)** 이면 ``settings.search_backend``
    (없으면 ``'pg'`` — 020 opt-in 폴백 동형, 필드 정식화는 G4)로 해소한다. ``'opensearch'`` **외엔
    전부 현 pg 경로**(``_grouped_fn`` 무변경 → 회귀 0·SC-001). ``'opensearch'`` 면 **text·audio·image·
    video 모든 버킷**을 020 OS 인덱스(nori BM25 캡션·라벨 + ``embedding`` kNN + 정규화 융합)에서
    ``_os_search_fn`` 으로 검색해 **같은 키**(text_documents·audio·image·video)로 반환한다(022 — image/
    video 를 더 이상 PG CLIP 으로 보내지 않음, FR-003·SC-005 응답 동형). OS 경로는 PG grouped(시각
    CLIP)·``structure_user_query``(LLM)·``structured`` 를 호출하지 않고 원문 ``query`` 를 쓴다(멀티모달
    LLM 0 — FR-002·SC-004). OS 미도달이면 ``_os_search_fn``/``_os_client_fn`` 예외를 **그대로 전파**한다
    (FR-007·SC-006 — silent pg 폴백 금지). ``_os_search_fn``/``_os_client_fn`` 은 테스트 주입 seam
    (기본 ``opensearch_search.search_assets_os``/``get_client``).
    """
    if modalities is not None:
        unknown = [m for m in modalities if m not in _MODALITY_BUCKETS]
        if unknown:
            raise ValueError(f"알 수 없는 모달리티: {unknown}")
        label_keys = [(m, _MODALITY_BUCKETS[m]) for m in modalities]
    else:
        label_keys = list(_MODALITY_BUCKETS.items())

    # 시각 2단계(CLIP)는 image·video 버킷에만 기여한다. 둘 다 요청하지 않았으면(text/audio 전용)
    # grouped 에 include_visual=False 를 넘겨 CLIP 경로를 통째로 건너뛴다 — 어차피 버려질 시각
    # 후보를 계산하지 않아 비용↓·결과 동치. 전체(None) 또는 image/video 포함이면 기존대로 True.
    include_visual = modalities is None or bool(set(modalities) & {"image", "video"})

    # 채널·질의모델 해소(018, FR-004). 명시 전달은 그대로 우선(A/B). 미지정(None)이면 운영 활성
    # 프로파일(적재·검색·관계 단일 출처)로 해소한다 — text_channel 은 active_embed_channel(),
    # 질의 모델은 (해소된) 채널의 model_for_channel 로 일치시킨다.
    # settings 미초기화(순수 단위 등)에서는 활성 해소가 RuntimeError 이므로 기존 기본('st', None)으로
    # 보수적 폴백한다 — query_model_name=None 은 media_search 가 기존대로 KoSimCSE 로 해소(회귀 0).
    query_model_name = text_query_model
    try:
        if text_channel is None:
            text_channel = active_embed_channel()
        if query_model_name is None:
            query_model_name = model_for_channel(text_channel)
    except RuntimeError:
        # settings 미초기화: 기존 검색 단위(006/017 기본 경로)가 settings 없이 그대로 동작.
        # 운영 진입점(run_search 등)은 항상 init_settings 하므로 이 폴백은 비운영(테스트) 경로다 —
        # 오설정(운영서 init_settings 누락)을 관측 가능하게 warning 으로 남긴다(동작 불변).
        _LOG.warning("settings 미초기화 — 활성 임베딩 채널 'st' 보수 폴백(운영은 init_settings 필수)")
        if text_channel is None:
            text_channel = EMBEDDING_KIND_ST
        # query_model_name 은 None 유지 → media_search 가 기존대로 cfg.text_embedding_model 로 해소.

    # 백엔드 해소(021, 020 opt-in 동형): backend 인자 우선, 미지정이면 settings.search_backend(없으면
    # 'pg'). search_backend 필드 정식화·화이트리스트 검증은 G4(T007) — 여기선 'opensearch' 외엔 전부
    # 현 pg 경로다. settings 미초기화(순수 단위 등)면 cfg=None → getattr 폴백으로 'pg'(회귀 0).
    try:
        cfg = get_current_settings()
    except RuntimeError:
        cfg = None
    backend_name = backend if backend is not None else getattr(cfg, "search_backend", "pg")

    # 024: per-result 적합도 임계는 backend 별 스케일이 다르다 — OS 정규화 융합 점수(min-max·top≈1.0)
    # 에 PG 코사인 스케일 min_scores(image 0.25 등)를 적용하면 사실상 무필터라 노이즈 꼬리가 남는다.
    # OS 경로는 settings 의 OS 전용 임계(search_os_min_scores, SEARCH_OS_MIN_SCORE_*)가 전달
    # min_scores 를 **대체**하고(명시 인자 우선 관례의 의도적 예외 — 운영 진입점이 PG 스케일 값을
    # 그대로 넘기기 때문), pg 경로·settings 미초기화(폴백)는 전달 min_scores 그대로 둔다(회귀 0 —
    # FR-002). 단 **명시 빈 dict({})는 "필터 비활성" 센티넬**로 존중한다 — no_cutoff 디버그 경로
    # (sample_search_api)가 OS 백엔드에서도 약한 후보까지 노출할 수 있어야 하므로(리뷰 후속).
    if backend_name == "opensearch" and min_scores != {}:
        effective_min_scores = getattr(cfg, "search_os_min_scores", None) or min_scores
    else:
        effective_min_scores = min_scores

    if backend_name == "opensearch":
        # 022: text·audio·image·video 전 모달리티를 OS(020 인덱스)로 검색(PG CLIP·LLM 미접촉).
        # OS 미도달 예외는 전파(FR-007). PG 전용 인자(structured·alpha·fusion·query_model_name·
        # chunk_agg·_grouped_fn)는 OS 경로 미사용이라 넘기지 않는다(아래 else pg 분기에서만 사용).
        grouped = _grouped_via_opensearch(
            query,
            modalities=modalities,
            limit_per_bucket=limit_per_bucket,
            text_channel=text_channel,
            cfg=cfg,
            os_search_fn=_os_search_fn,
            os_client_fn=_os_client_fn,
        )
    else:
        # backend != 'opensearch'(기본 pg): 기존 코드 경로 그대로 — 한 줄도 바꾸지 않는다(회귀 0·SC-001).
        grouped = _grouped_fn(
            query,
            structured=structured,
            limit_per_bucket=limit_per_bucket,
            text_hybrid_alpha=text_hybrid_alpha,
            image_search_alpha=image_search_alpha,
            fusion=fusion,
            channel=text_channel,
            query_model_name=query_model_name,
            chunk_agg=chunk_agg,
            include_visual=include_visual,
        )
    # per-result 적합도 필터: backend 별 스케일로 해소된 effective_min_scores 를 적용한다(024).
    # OS=정규화 점수 스케일(search_os_min_scores)·pg=전달 min_scores(PG 코사인 스케일) — 위 backend
    # 분기에서 결정. 0.0(미설정)이면 비활성(_filter_by_min_score 원본 반환).
    results = {
        key: _filter_by_min_score(grouped.get(key, []), (effective_min_scores or {}).get(label, 0.0))
        for label, key in label_keys
    }
    return {"query": query, "results": results, "meta": grouped.get("meta", {})}
