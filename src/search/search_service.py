"""하이브리드 검색 서비스 진입점 — 호출부(CLI/HTTP)에 독립적인 함수 계층.

요청(query·modalities·limit·alpha)을 받아 ``search_media_all_grouped`` 로 모달리티별
버킷 결과를 만든 뒤, 요청한 모달리티만 골라 일정한 모양으로 반환한다. 실제 검색·LLM·DB는
``media_search`` 가 담당하고, 본 모듈은 요청 정규화·필터·응답 형태만 책임진다(F-4.3, 단계 C+).
"""

from __future__ import annotations

from typing import Any, Callable

from src.search.media_search import search_media_all_grouped

# 요청 모달리티 라벨 → ``search_media_all_grouped`` 결과 버킷 키.
_MODALITY_BUCKETS: dict[str, str] = {
    "text": "text_documents",
    "audio": "audio",
    "image": "image",
    "video": "video",
}


def search_hybrid(
    query: str,
    *,
    modalities: list[str] | None = None,
    limit_per_bucket: int = 20,
    text_hybrid_alpha: float = 0.75,
    image_search_alpha: float = 0.65,
    _grouped_fn: Callable[..., dict[str, Any]] = search_media_all_grouped,
) -> dict[str, Any]:
    """질의를 하이브리드 검색해 모달리티 버킷으로 반환한다.

    ``modalities`` 가 ``None`` 이면 전체 버킷(text/audio/image/video)을, 지정하면 해당
    버킷만 반환한다. 알 수 없는 모달리티 라벨은 ``ValueError``.
    ``_grouped_fn`` 은 테스트 주입 seam(미주입=실제 ``search_media_all_grouped``).
    """
    if modalities is not None:
        unknown = [m for m in modalities if m not in _MODALITY_BUCKETS]
        if unknown:
            raise ValueError(f"알 수 없는 모달리티: {unknown}")
        wanted = [_MODALITY_BUCKETS[m] for m in modalities]
    else:
        wanted = list(_MODALITY_BUCKETS.values())

    grouped = _grouped_fn(
        query,
        limit_per_bucket=limit_per_bucket,
        text_hybrid_alpha=text_hybrid_alpha,
        image_search_alpha=image_search_alpha,
    )
    results = {key: grouped.get(key, []) for key in wanted}
    return {"query": query, "results": results, "meta": grouped.get("meta", {})}
