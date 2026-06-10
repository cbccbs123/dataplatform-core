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
from collections.abc import Callable, Iterable
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


# ──────────────────────────────────────────────────────────────────────────
# IO 함수 (G2) — opensearch-py·임베더는 모듈 상단이 아니라 **함수 내부에서 지연 import**.
# 플래그 off(pg 백엔드) 환경의 모듈 순수성(상단 import 없음)을 보존하기 위함이다 — 위 순수 함수만
# 쓰는 단위 게이트는 opensearch-py·임베더 미설치여도 import 가능해야 한다(020 동형).
# 실제 OS 동작 검증은 G5(실OS e2e). 여기 IO 는 가짜 OS 클라이언트로 액션 조립을 단위 검증한다.
# ──────────────────────────────────────────────────────────────────────────


def search_pipeline_body(weights: tuple[float, float]) -> dict[str, Any]:
    """normalization-processor 검색 파이프라인 본문을 만든다(순수·결정적, FR-005).

    BM25(텍스트 서브쿼리)·kNN 점수를 **min-max 정규화 후 가중평균**(arithmetic_mean)으로 융합한다.
    ``weights`` 는 ``build_search_body`` 의 hybrid 서브쿼리 순서 ``[텍스트(BM25), knn]`` 와 **동일
    순서**의 가중치 ``(w_bm25, w_knn)`` 다 — 측정 프로토타입·OS 네이티브 방식과 일치(plan §R2).
    """
    w_bm25, w_knn = weights
    return {
        "description": "021 하이브리드 검색 정규화 융합(min-max + 가중평균) — BM25 ⊕ kNN",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [float(w_bm25), float(w_knn)]},
                    },
                }
            }
        ],
    }


def ensure_search_pipeline(
    client: Any, name: str, *, weights: tuple[float, float] = (0.5, 0.5)
) -> str:
    """정규화 검색 파이프라인이 없으면 등록한다(멱등 — 020 ``ensure_index`` 동형, FR-006).

    존재 여부는 ``client.search_pipeline.get()``(id 미지정 → 전체 파이프라인 dict, 없으면 ``{}``)
    의 멤버십으로 판단한다(020 ``indices.exists`` boolean-멱등과 동형). 부재면 ``put`` 으로 1회
    등록하고, 존재하면 no-op 이다(재실행 안전). 반환: ``'created'`` | ``'exists'``.

    ⚠️ 실 opensearch-py 검색 파이프라인 API(``client.search_pipeline.get/put``)는 best-effort 로
    맞췄다(opensearch-py 3.x ``SearchPipelineClient.get(id=...)``·``put(id=, body=)``). 실 OS 응답
    형태·존재 시맨틱은 G5 실OS e2e 에서 확정 검증한다.
    """
    existing = client.search_pipeline.get() or {}
    if name in existing:
        return "exists"
    client.search_pipeline.put(id=name, body=search_pipeline_body(weights))
    return "created"


def embed_query(query: str, *, channel: str) -> list[float]:
    """질의를 인덱스와 **동일 채널·모델**로 임베딩한다(FR-004 질의-문서 일치, 018 단일 출처).

    적재·검색이 공유하는 텍스트 임베더(`media_search.embed_query_for_media_search`)를 **재사용**한다
    — 임베딩 로직을 복제하지 않는다. 채널→모델 해소는 단일 출처 ``settings.model_for_channel``
    (017 A/B)로 한다(기본 ``'st'`` → KoSimCSE, ``'st_bge'`` → BGE-M3). 020 인덱스가 활성 채널
    임베딩을 색인하므로, 질의도 같은 채널 모델로 임베딩해야 같은 벡터 공간에서 비교된다.
    """
    from src.config.settings import model_for_channel
    from src.search.media_search import embed_query_for_media_search

    return embed_query_for_media_search(query, model_name=model_for_channel(channel))


def search_assets_os(
    client: Any,
    query: str,
    *,
    modalities: Iterable[str],
    k: int = 100,
    channel: str = "st",
    weights: tuple[float, float] = (0.5, 0.5),
    index: str,
    pipeline_name: str,
    exclude_medical: bool = True,
    embed_fn: Callable[..., list[float]] = embed_query,
) -> dict[str, list[dict[str, Any]]]:
    """text·audio 버킷을 020 OS 인덱스에서 하이브리드 검색한다(FR-002 — OS 경로 모달리티).

    흐름: (a) ``embed_fn`` 으로 질의를 **1회만** 임베딩해 벡터를 modality 간 재사용한다(질의-문서
    일치·중복 임베딩 방지), (b) modality 마다 ``build_search_body`` 로 하이브리드 본문을 만들어
    ``client.search(index=, body=, search_pipeline=pipeline_name)`` 으로 정규화 융합 검색하고,
    (c) hit 들을 ``os_hit_to_row`` 로 매핑해 ``{modality: [rows]}`` 버킷으로 돌려준다(SC-005 동형).

    OS 미도달(``client.search`` 예외)이면 **그대로 전파**한다(FR-008 — silent pg 폴백 금지: 결과가
    백엔드 가용성에 따라 달라지면 결정성·관측성 훼손). 검색은 PG·OS 를 **읽기 전용**으로만 만진다
    (헌법 6조). 검색용 LLM 질의 구조화는 호출하지 않는다(FR-004 — 원문 질의를 nori·임베딩에 직접).
    """
    query_vector = embed_fn(query, channel=channel)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for modality in modalities:
        body = build_search_body(
            query,
            query_vector,
            modality=modality,
            k=k,
            weights=weights,
            exclude_medical=exclude_medical,
        )
        resp = client.search(index=index, body=body, search_pipeline=pipeline_name)
        hits = ((resp or {}).get("hits") or {}).get("hits") or []
        buckets[modality] = [os_hit_to_row(h) for h in hits]
    return buckets


def get_client(url: str | None = None) -> Any:
    """검색용 OpenSearch 클라이언트 — 020 ``opensearch_sync.get_client`` 재사용(단일 출처)."""
    from src.search.opensearch_sync import get_client as _sync_get_client

    return _sync_get_client(url)
