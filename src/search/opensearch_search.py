"""OpenSearch 검색 쿼리 빌더·결과 매핑 (검색 read path → OpenSearch, spec 021 G1).

020 이 깐 단일 인덱스(nori 한국어 BM25 + ``knn_vector``)를 **읽기만** 한다(쓰기 0, 헌법 6조).
본 모듈의 **순수 함수**(`build_search_body`·`os_hit_to_row`)는 OS·DB·opensearch-py 없이
결정적으로 동작하며 단위 게이트에서 항상 검증된다. 실제 검색 실행(IO: 검색 파이프라인 등록·
질의 임베딩·search_assets_os)은 후속 그룹(G2)에서 추가하며, opensearch-py 의존은 모듈 상단이
아니라 해당 함수 내부에서 지연 import 한다(플래그 off 환경의 순수성 보존 — 020 동형).

필드명 정본 = 020 인덱스 매핑(`opensearch_sync.build_index_body`):
    - nori 텍스트: ``summary``·``keywords``·``labels``·``file_name``·``search_text``
    - 벡터: ``embedding``(knn_vector·1536D)
    - keyword 필터: ``modality``·``domain_label``
"""

from __future__ import annotations

import math
from typing import Any

# 020 인덱스의 nori 텍스트 필드(BM25 multi_match 대상). 필드명 정본 = opensearch_sync.build_index_body.
# 주의: labels 는 매핑상 keyword 지만 plan §1 이 multi_match 대상에 포함한다 — multi_match 는 keyword
# 필드를 정확매칭 절로 안전 수용한다(텍스트 필드와 혼합 무해). text/audio 버킷 한국어 BM25 재현율용.
_TEXT_FIELDS: tuple[str, ...] = (
    "summary",
    "keywords",
    "labels",
    "file_name",
    "search_text",
)

# FR-011(헌법 10조 · 010 FR-014): 의료 자산은 검색 결과에서 제외(domain_label keyword 필터).
_MEDICAL_LABEL = "medical"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """None·비수치·NaN·inf 를 안전한 유한 실수로 정규화(결정적·순수)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def build_search_body(
    query: str,
    query_vector: list[float],
    *,
    modality: str,
    k: int = 100,
    weights: tuple[float, float] = (0.5, 0.5),
    exclude_medical: bool = True,
) -> dict[str, Any]:
    """OpenSearch 하이브리드 검색 본문을 만든다(순수·결정적).

    nori 텍스트 ``multi_match``(BM25) ⊕ ``knn``(embedding, 질의 벡터)의 **hybrid 쿼리**다. 두
    서브쿼리는 각각 ``bool`` 로 감싸 ① ``modality`` keyword term 필터, ② ``exclude_medical`` 이면
    ``domain_label='medical'`` ``must_not``(FR-011)을 균일 적용한다.

    점수 융합(BM25·kNN 의 min-max 정규화 + 가중평균)은 OpenSearch **검색 파이프라인**
    (normalization-processor, G2 ``ensure_search_pipeline``)이 담당하므로 본문은 hybrid 서브쿼리
    구성까지만 책임진다 — ``weights`` 는 그 파이프라인 등록용 융합 가중치 메타로, 쿼리 본문에는
    반영하지 않는다(호출부 search_assets_os 가 파이프라인에 전달, G2).

    정렬은 점수 내림차순 + **동점 ``asset_id`` 오름차순**(FR-009 결정적 tiebreaker, 헌법 3조).
    ``size`` 는 ``k``. 필드명·차원은 020 인덱스 매핑(`opensearch_sync.build_index_body`)과 일치.
    """
    filters: list[dict[str, Any]] = [{"term": {"modality": modality}}]
    must_not: list[dict[str, Any]] = []
    if exclude_medical:
        must_not.append({"term": {"domain_label": _MEDICAL_LABEL}})

    def _wrap(inner: dict[str, Any]) -> dict[str, Any]:
        # 각 서브쿼리를 bool 로 감싸 같은 modality 필터·의료배제를 균일 적용한다.
        clause: dict[str, Any] = {"must": [inner], "filter": list(filters)}
        if must_not:
            clause["must_not"] = list(must_not)
        return {"bool": clause}

    text_sub = _wrap({"multi_match": {"query": query, "fields": list(_TEXT_FIELDS)}})
    knn_sub = _wrap({"knn": {"embedding": {"vector": list(query_vector), "k": int(k)}}})

    return {
        "size": int(k),
        "query": {"hybrid": {"queries": [text_sub, knn_sub]}},
        # 결정적 tiebreaker(FR-009): 정규화 점수 desc → 동점 asset_id(keyword) asc.
        "sort": [{"_score": {"order": "desc"}}, {"asset_id": {"order": "asc"}}],
    }


def os_hit_to_row(hit: dict[str, Any]) -> dict[str, Any]:
    """OpenSearch hit 을 media_search 버킷 행과 동형(SC-005)인 dict 로 매핑한다(순수·결정적).

    media_search 버킷 행의 핵심 키(``id``·``file_uri``·``modality``·``summary``·``similarity``)에
    맞춘다 — 검색 서비스(G3)가 OS 버킷(text·audio)과 PG 버킷(image·video)을 같은 모양으로
    병합할 수 있게 한다(응답 동형). ``similarity`` 는 hit 의 ``_score``(검색 파이프라인이 BM25·
    kNN 을 min-max 정규화·가중평균한 융합 점수)다. ``_source`` 누락·메타 None 도 안전 처리한다.

    ⚠️ media_search 버킷 행의 자산 식별 키는 ``asset_id`` 가 아니라 ``id`` 다(검색 SQL alias
    ``asset_id AS id``). 따라서 020 인덱스 ``_source.asset_id``(== ``_id``)를 이 ``id`` 로 옮긴다.
    """
    src = hit.get("_source") or {}
    asset_id = src.get("asset_id") or hit.get("_id")
    return {
        "id": str(asset_id) if asset_id is not None else None,
        "file_uri": str(src.get("fs_uri") or ""),
        "modality": src.get("modality"),
        "summary": str(src.get("summary") or ""),
        "similarity": _safe_float(hit.get("_score"), 0.0),
    }
