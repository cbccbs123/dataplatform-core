"""하이브리드 검색 서비스 진입점 — 호출부(CLI/HTTP)에 독립적인 함수 계층.

요청(query·modalities·limit·alpha)을 받아 ``search_media_all_grouped`` 로 모달리티별
버킷 결과를 만든 뒤, 요청한 모달리티만 골라 일정한 모양으로 반환한다. 실제 검색·LLM·DB는
``media_search`` 가 담당하고, 본 모듈은 요청 정규화·필터·응답 형태만 책임진다(F-4.3, 단계 C+).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from src.config.settings import model_for_channel
from src.search.media_search import EMBEDDING_KIND_ST, search_media_all_grouped

# 요청 모달리티 라벨 → ``search_media_all_grouped`` 결과 버킷 키.
# 결정성(헌법 3조): 결과 버킷 조립이 ``list(.items())`` 순회 순서에 의존하므로 삽입 순서를
# 보존한다(dict 는 3.7+ 삽입 순서 보장). set 등 순서 비보장 타입으로 대체 금지.
_MODALITY_BUCKETS: dict[str, str] = {
    "text": "text_documents",
    "audio": "audio",
    "image": "image",
    "video": "video",
}


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
    text_channel: str = EMBEDDING_KIND_ST,
    text_query_model: str | None = None,
    _grouped_fn: Callable[..., dict[str, Any]] = search_media_all_grouped,
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

    ``text_channel``/``text_query_model`` 은 017 A/B 텍스트 임베딩 채널 선택이다(텍스트 채널 한정,
    시각 CLIP 경로 무변경). 기본값 ``('st', None)`` 이면 기존 동작과 완전 동치 — grouped 에
    ``channel='st'``·``query_model_name=None`` 을 넘기고, 질의 임베딩 모델 해소는 media_search 가
    기존대로(``cfg.text_embedding_model``=KoSimCSE) 맡는다(이 기본 경로는 ``get_current_settings``
    를 건드리지 않아 settings 미초기화 환경에서도 동작 보존, 회귀 0). ``text_query_model`` 을 주면
    그 모델을 그대로 쓰고, 미지정이면서 ``text_channel`` 이 기본('st')이 아니면 ``model_for_channel``
    로 해소한다(FR-004 질의-문서 모델 일치; 미지원 채널은 ``ValueError``).
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

    # 질의 임베딩 모델 해소(FR-004): 명시 모델이 우선. 미지정이면서 기본 채널('st')이면
    # None 을 그대로 넘겨 media_search 가 기존 KoSimCSE 경로로 해소하게 둔다 — 기본 경로는
    # get_current_settings 를 건드리지 않아 settings 미초기화에서도 동작 보존(회귀 0). 비-기본
    # 채널(예: 'st_bge')은 A/B 에 설정이 필수이므로 model_for_channel 로 즉시 해소한다.
    query_model_name = text_query_model
    if query_model_name is None and text_channel != EMBEDDING_KIND_ST:
        query_model_name = model_for_channel(text_channel)

    grouped = _grouped_fn(
        query,
        structured=structured,
        limit_per_bucket=limit_per_bucket,
        text_hybrid_alpha=text_hybrid_alpha,
        image_search_alpha=image_search_alpha,
        fusion=fusion,
        channel=text_channel,
        query_model_name=query_model_name,
        include_visual=include_visual,
    )
    results = {
        key: _filter_by_min_score(grouped.get(key, []), (min_scores or {}).get(label, 0.0))
        for label, key in label_keys
    }
    return {"query": query, "results": results, "meta": grouped.get("meta", {})}
