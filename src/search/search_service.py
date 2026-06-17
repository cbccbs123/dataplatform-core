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

from src.config import search_constants
from src.config.embedding_constants import EMBEDDING_KIND_ST
from src.config.settings import (
    ChunkAggConfig,
    active_embed_channel,
    get_current_settings,
    model_for_channel,
)
from src.search.media_search import search_media_all_grouped

# 021 G3: OpenSearch 백엔드 분기용 seam. opensearch_search 모듈 상단은 순수(opensearch-py·임베더는
# 함수 내부 지연 import)라 pg 기본 환경에서도 import 안전 — 실제 OS IO 는 backend='opensearch' 호출
# 시에만 발생한다(플래그 off 순수성 보존).
from src.search.opensearch_search import get_client as os_get_client
from src.search.opensearch_search import normalize_query as os_normalize_query
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

# 027: OS 융합·게이트·컷 기본값은 모듈 중복 상수(_DEFAULT_OS_*)를 두지 않고 src.config.search_constants
# 단일 출처(F1)를 직접 참조한다 — settings 미초기화(순수 단위 등) getattr 폴백도 같은 공개 상수를 쓴다
# (cross-module private import 없음·하드코딩 0). 미초기화 시 게이트 기본은 운영 기본과 동일(enabled True)
# 이며, search_backend='pg'(기본)면 OS 경로 자체가 미실행이라 회귀 0(SC-001).


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
    disable_os_cutoff: bool,
    os_search_fn: Callable[..., tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]],
    os_client_fn: Callable[..., Any],
    query_norm_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """backend='opensearch' 경로의 모달리티 버킷을 조립한다(022·027, FR-002·FR-003·SC-005).

    text·audio·**image·video 모든 버킷**을 020 OS 인덱스에서 동일 하이브리드(nori BM25 캡션·라벨 +
    ``embedding`` kNN + **클라이언트 융합**)로 검색한다 — image/video 도 020 assets 인덱스에 한국어 VLM
    캡션·KoSimCSE 캡션 임베딩으로 색인돼 있어 text/audio 와 같은 경로다(CLIP 아님; 시각-내용 매칭은 후속).
    요청 모달리티 전체를 **한 번의** ``os_search_fn`` 호출로 검색해 버킷을 만들고, 현 pg 분기와 같은
    키(text_documents·audio·image·video·meta)로 담는다 — 반환 grouped 는 pg 분기와 동일 모양이라 호출부의
    ``_filter_by_min_score`` 공유 코드가 그대로 처리한다(응답 동형).

    설계 판단:
    - **PG·LLM 미접촉(FR-002·SC-004)**: PG grouped(시각 CLIP 2단계)·``structure_user_query``(검색 질의
      구조화 LLM)를 호출하지 않는다 — 멀티모달 LLM 0·ms. 따라서 PG 전용 파라미터(structured·alpha·
      fusion·query_model_name·chunk_agg·grouped_fn)는 이 경로에서 제거됐다.
    - **query-norm 토글(029 FR-004·021 개정)**: ``search_os_query_norm_enabled`` on(기본 off)이면 검색
      직전 질의를 **명사구 정규화**(``noun_phrase_query``·gemma·temp=0·env 입력 0·src/llm/client 단일
      seam)한 뒤 그 질의를 OS 에 넘긴다 — 정규화를 service 레벨에서 1회만 수행해(중복 LLM 0) 정규화된
      질의가 OS seam 안에서 임베딩·BM25·rerank 채점에 동일 적용되게 한다. 관측성은 top-level
      ``meta["query_norm"]`` 로 노출(os_gate 미오염). off 면 원문 passthrough(바이트 동일·noun_phrase
      미호출 — 021 FR-004 기본 동작 보존, SC-001). ``query_norm_fn`` 은 테스트 주입 seam.
    - **(buckets, gate_meta) 튜플 수신(027)**: ``os_search_fn``(search_assets_os)은 클라이언트 융합
      전환으로 버킷과 함께 게이트 메타(모달리티별 top·baseline·gate_passed·cut_count)를 돌려준다 →
      ``meta["os_gate"]`` 로 합류시켜 빈 버킷이 no-match 판정인지 즉시 관측 가능하게 한다(F4 관측성).
    - **모달리티 키 매핑**: ``os_search_fn`` 버킷 키는 모달리티명('text'/'image')이고 응답 grouped 키는
      ('text_documents'/'image')이므로 ``_MODALITY_BUCKETS`` 로 변환해 담는다.
    - **컷오프 설정(027)**: 게이트·per-result 컷 임계(eps·floor·result_floor·operator)를 cfg 에서 읽어
      OS seam 에 전달한다(getattr 폴백은 search_constants 단일 출처 — settings 미초기화 순수 단위 방어).
      ``disable_os_cutoff=True`` 면 ``cutoff_enabled=False`` 로 강제해 게이트·per-result 컷을 모두 끈다
      (no_cutoff 디버그 우회 — 약한 후보까지 노출). per-result 컷이 search_assets_os 내부 코사인 스케일
      (cut_rows·result_floor)로 이동했으므로 호출부 ``_filter_by_min_score`` 는 OS 경로에 적용하지 않는다.
    - **OS 미도달(FR-007)**: ``os_client_fn``/``os_search_fn`` 예외를 try/except 로 감싸지 않아 그대로
      전파한다(silent pg 폴백 금지 — 결과가 백엔드 가용성에 따라 달라지지 않게).

    ⚠️ 결정성(헌법 3조): 최종 응답 버킷 순서는 호출부의 ``label_keys`` 가 정하므로 여기 grouped 의
    삽입 순서는 출력 순서에 영향하지 않는다.
    """
    requested = modalities if modalities is not None else list(_MODALITY_BUCKETS)

    if not requested:
        # 빈 요청: OS 미접촉(불필요 IO 회피). os_gate 도 빈 dict(검색 안 함).
        return {"meta": {"backend": "opensearch", "os_gate": {}}}

    # disable_os_cutoff(디버그 우회)면 게이트·컷 모두 off. 아니면 cfg 의 enabled(미초기화 폴백=운영 기본).
    cutoff_enabled = (
        False
        if disable_os_cutoff
        else getattr(cfg, "search_os_cutoff_enabled", search_constants.OS_CUTOFF_ENABLED_DEFAULT)
    )

    # 029 query-norm(021 FR-004 토글 개정): cfg 토글(getattr 폴백=search_constants 단일 출처·기본 off)을
    # 읽어, on 이면 검색 직전 질의를 **service 레벨에서 1회** 명사구 정규화한다. 정규화를 여기 한 곳에서
    # 끝내는 이유: ① LLM 호출을 1회로(중복 0), ② 관측성(FR-007)을 top-level meta["query_norm"] 로 노출해
    # 모달리티 키 dict 인 os_gate(gate_meta)를 오염시키지 않음(골든 하니스 등 gate_meta 순회 소비자 보호 —
    # search_assets_os 는 별도 반환·gate_meta 오염 없이 정규화된 질의만 받음). off(기본)면 normalize_query
    # 가 원문 그대로 돌려줘 noun_phrase_query 미호출·바이트 동일(SC-001·FR-008). 정규화된 질의가 OS seam
    # 안에서 임베딩·BM25·rerank 채점에 동일 적용된다(query_norm_fn 미주입+enabled 면 noun_phrase_query 지연 import).
    qn_enabled = getattr(
        cfg, "search_os_query_norm_enabled", search_constants.OS_QUERY_NORM_ENABLED_DEFAULT
    )
    norm_fn = query_norm_fn
    if qn_enabled and norm_fn is None:
        from src.search.query_preprocess import noun_phrase_query as norm_fn
    os_query = os_normalize_query(query, enabled=qn_enabled, llm_fn=norm_fn)

    client = os_client_fn()  # OS 클라이언트 생성 실패 시 예외 전파(FR-007)
    os_buckets, gate_meta = os_search_fn(
        client,
        os_query,  # 029: 정규화된 질의(off 면 원문 그대로) — 임베딩·BM25·rerank 채점에 동일 적용
        modalities=requested,  # 요청 전 모달리티(image/video 포함)를 한 번에 OS 검색
        k=limit_per_bucket,
        channel=text_channel,
        weights=getattr(cfg, "opensearch_fusion_weights", search_constants.OS_FUSION_WEIGHTS_DEFAULT),
        index=getattr(cfg, "opensearch_index", "assets"),
        exclude_medical=True,
        # 027: 버킷 게이트 + per-result 컷 임계를 cfg 에서 읽어 OS seam 에 전달한다(getattr 폴백은
        # search_constants 단일 출처 — settings 미초기화 순수 단위 방어). disable_os_cutoff 면 위에서
        # cutoff_enabled=False 로 강제돼 search_assets_os 가 게이트·컷 모두 끄고 융합 전체를 노출한다.
        cutoff_enabled=cutoff_enabled,
        cutoff_eps=getattr(cfg, "search_os_cutoff_eps", search_constants.OS_CUTOFF_EPS_DEFAULT),
        cutoff_floor=getattr(cfg, "search_os_cutoff_floor", search_constants.OS_CUTOFF_FLOOR_DEFAULT),
        result_floor=getattr(cfg, "search_os_result_floor", search_constants.OS_RESULT_FLOOR_DEFAULT),
        # 025: BM25 operator — 'and' 면 전 토큰 매칭(F2 복합어 가짜매칭 차단). 미초기화 폴백 'or'(현행).
        bm25_operator=getattr(cfg, "search_os_bm25_operator", search_constants.OS_BM25_OPERATOR_DEFAULT),
        # 028: rerank 평가 설정(기본 off — 회귀 0). on 이면 게이트·컷을 대체하는 평가 경로.
        rerank_enabled=getattr(cfg, "search_os_rerank_enabled", search_constants.OS_RERANK_ENABLED_DEFAULT),
        rerank_top_r=getattr(cfg, "search_os_rerank_top_r", search_constants.OS_RERANK_TOP_R_DEFAULT),
        rerank_tau=getattr(cfg, "search_os_rerank_tau", search_constants.OS_RERANK_TAU_DEFAULT),
        rerank_model=getattr(cfg, "search_os_rerank_model", search_constants.OS_RERANK_MODEL_DEFAULT),
    )  # client.msearch 미도달 예외도 전파(FR-007)
    # meta 에 게이트 관측성(os_gate) 합류 — 빈 버킷이 no-match 판정인지 즉시 확인(F4).
    grouped: dict[str, Any] = {"meta": {"backend": "opensearch", "os_gate": gate_meta}}
    # 029 query-norm 관측성(FR-007): on 일 때만 top-level meta["query_norm"] 로 원문→정규화 매핑을 노출
    # 한다(os_gate 는 모달리티 키 dict 이라 오염 금지). off(기본)면 키 자체를 두지 않아 027 meta 와 바이트
    # 동일(SC-001 — 기존 meta 형태 봉인 테스트 무영향).
    if qn_enabled:
        grouped["meta"]["query_norm"] = {
            "enabled": True, "original": query, "normalized": os_query,
        }
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
    disable_os_cutoff: bool = False,
    text_channel: str | None = None,
    text_query_model: str | None = None,
    chunk_agg: ChunkAggConfig | None = None,
    backend: str | None = None,
    _grouped_fn: Callable[..., dict[str, Any]] = search_media_all_grouped,
    _os_search_fn: Callable[
        ..., tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]
    ] = os_search_assets,
    _os_client_fn: Callable[..., Any] = os_get_client,
    _query_norm_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """질의를 하이브리드 검색해 모달리티 버킷으로 반환한다.

    ``modalities`` 가 ``None`` 이면 전체 버킷(text/audio/image/video)을, 지정하면 해당
    버킷만 반환한다. 알 수 없는 모달리티 라벨은 ``ValueError``. 요청에 image·video 가
    하나도 없으면(text/audio 전용) grouped 에 ``include_visual=False`` 를 넘겨 시각 2단계
    (CLIP)를 건너뛴다 — 버려질 시각 후보를 계산하지 않아 비용↓, 반환 버킷은 동일.
    ``structured`` 를 넘기면 그대로 grouped 검색에 전달돼 LLM 질의 구조화를 건너뛴다
    (이미 구조화됐거나 LLM 없이 테스트할 때). ``_grouped_fn`` 은 테스트 주입 seam.
    ``min_scores`` 는 모달리티 라벨→적합도 하한(0.0=비활성); 각 버킷에서 ``similarity`` 가
    임계값 미만인 행을 응답에서 제외한다(미지정 모달리티는 필터하지 않음). ⚠️ **pg 경로 전용**이다 —
    OS 경로(backend='opensearch')는 per-result 컷이 ``search_assets_os`` 내부 코사인 스케일
    (``cut_rows``·``result_floor``)로 이동했으므로 전달 ``min_scores``(PG 코사인 스케일)를 적용하지
    않는다(스케일 불일치 방지·F1). pg 경로는 종전대로 전달 ``min_scores`` 그대로(회귀 0).
    ``disable_os_cutoff`` 는 OS 경로의 **게이트·per-result 컷을 모두 끄는 디버그 우회**다(기본 False).
    True 면 ``cutoff_enabled=False`` 로 전달돼 약한 후보까지 노출한다(sample_search_api ``no_cutoff``
    배선). pg 경로엔 무영향.
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
    CLIP)·``structure_user_query``(검색 질의 구조화 LLM)·``structured`` 를 호출하지 않는다(멀티모달 LLM
    0 — FR-002·SC-004). 다만 **029 query-norm 토글**(``search_os_query_norm_enabled``, 기본 off)이 on
    이면 검색 직전 질의를 명사구 정규화(단일 seam·temp=0·env 입력 0)한 뒤 임베딩·BM25·rerank 채점에
    동일 적용한다(021 FR-004 토글 개정) — off 면 원문 ``query`` 그대로(바이트 동일). OS 미도달이면
    ``_os_search_fn``/``_os_client_fn`` 예외를 **그대로 전파**한다(FR-007·SC-006 — silent pg 폴백 금지).
    ``_os_search_fn``/``_os_client_fn`` 은 테스트 주입 seam(기본 ``opensearch_search.search_assets_os``/
    ``get_client``). ``_query_norm_fn`` 은 query-norm seam(미주입+on 이면 ``noun_phrase_query`` 지연 import).
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

    # 027: per-result 적합도 컷은 backend 별로 다른 곳에서 한다. OS 경로는 search_assets_os 내부 코사인
    # 스케일(cut_rows·result_floor)이 노이즈 꼬리를 거르므로 호출부 _filter_by_min_score 를 적용하지
    # 않는다(전달 min_scores 는 PG 코사인 스케일이라 OS 정규화·코사인 점수에 무의미·스케일 불일치).
    # pg 경로만 전달 min_scores 그대로 적용한다(회귀 0 — FR-002). 024 의 effective_min_scores 분기·
    # 빈 dict 센티넬은 제거됐다(OS 컷이 seam 내부로 이동·디버그 우회는 disable_os_cutoff 가 담당).
    filter_min_scores = {} if backend_name == "opensearch" else min_scores

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
            disable_os_cutoff=disable_os_cutoff,
            os_search_fn=_os_search_fn,
            os_client_fn=_os_client_fn,
            query_norm_fn=_query_norm_fn,
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
    # per-result 적합도 필터(027): pg 경로만 전달 min_scores 를 적용한다(PG 코사인 스케일). OS 경로는
    # filter_min_scores={} 라 무필터(컷은 search_assets_os 내부에서 이미 수행). 0.0(미설정/빈)이면
    # 비활성(_filter_by_min_score 원본 반환).
    results = {
        key: _filter_by_min_score(grouped.get(key, []), (filter_min_scores or {}).get(label, 0.0))
        for label, key in label_keys
    }
    return {"query": query, "results": results, "meta": grouped.get("meta", {})}
