"""OpenSearch 서브검색 **본문 빌더** (BM25·kNN) + 모달리티/필드 상수 (검색 read path, spec 021·027·044·057).

069 US-E(FR-E5): 종전 ``opensearch_search`` 한 파일(융합 수학 + 본문 빌더 + IO)에서 **본문 빌더**만
분리했다. 여기의 함수는 OS·opensearch-py 없이 **순수·결정적**으로 body dict 를 만든다(단위 검증 가능·
헌법 3조). IO(실행)는 ``opensearch_search``, 융합·게이트 수학은 ``fusion`` 이 담당한다.

필드명 정본 = 020 인덱스 매핑(``opensearch_sync.build_index_body``): nori 텍스트(summary·keywords·
labels·file_name) + 벡터(embedding·1536D) + keyword 필터(modality·domain_label).

하위호환: 기존 ``opensearch_search.<name>`` import·patch 경로는 그 모듈이 이 심볼들을 **재export**해
그대로 유지된다(US-E patch seam 보존).
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from src.search.search_filters import SearchFilters, filters_to_opensearch_bool

# 020 인덱스의 nori 텍스트 필드(BM25 multi_match 대상). 필드명 정본 = opensearch_sync.build_index_body.
# 주의: labels 는 매핑상 keyword 지만 plan §1 이 multi_match 대상에 포함한다 — multi_match 는 keyword
# 필드를 정확매칭 절로 안전 수용한다(텍스트 필드와 혼합 무해). text/audio 버킷 한국어 BM25 재현율용.
# 026 T008(FR-003①): boost 차등 표기('field^N') — 토픽 신호(summary^3·keywords^2·labels^1)를
# 파일명(file_name^0.5)보다 강하게 둬서 파일명 노이즈가 랭킹을 압도하지 못하게 한다(F8 구조 방어).
# 047: 교차 필드 AND(recall) — ``multi_match`` ``cross_fields`` on summary+keywords(색인 search_text 제거).
_CROSS_META_FIELDS: tuple[str, ...] = ("summary^3", "keywords^2")

# 044 FR-101 · 047: named query _name 고정 집합(build_bm25_body bool.should).
BM25_NAMED_QUERY_NAMES: tuple[str, ...] = (
    "hit_keywords",
    "hit_labels",
    "hit_file_name",
    "hit_summary",
    "hit_cross_meta",
)

# FR-011(헌법 10조 · 010 FR-014): 의료 자산 검색 제외용 라벨(domain_label keyword 필터) — exclude_medical=True 일 때만.
# 2026-07-23 도메인 제외 전면 제거로 기본 OFF(의료 복귀 시 재도입).
_MEDICAL_LABEL = "medical"

# OS 검색 버킷 라벨 → 저장된 modality 값 집합. 요청 라벨('text')을 저장 modality 값으로 매핑한다.
# 053 canonical 전환: 저장 modality 는 이제 canonical('text'). 다만 재색인 완료 전까지 구 OS 문서엔
# file_kind 값(txt·json·pdf·office)이 남아 있을 수 있어, text 버킷을 **합집합**
# ({text} ∪ ALLOWED_TEXT_META_FILE_KINDS)으로 두어 구·신 문서를 동시 매칭한다(무중단·C5).
# 재색인 안정 후 frozenset({"text"})로 정리(FR-403). 단일 term 이 아니라 terms(집합) 필터를 써야 회수된다.
# 022 G1: image·video 는 020 assets 인덱스에 한국어 VLM 캡션(nori) + 활성 텍스트 채널 임베딩(현행
# st_api·bge-m3)으로 이미 색인돼 text/audio 와 동일 OS 하이브리드로 검색한다(CLIP 아님). 저장값=
# 라벨이라 단일값·문서화용 명시.
_MODALITY_VALUES: dict[str, frozenset[str]] = {
    "text": frozenset({"text", *ALLOWED_TEXT_META_FILE_KINDS}),
    "audio": frozenset({MediaKind.AUDIO.value}),
    MediaKind.IMAGE.value: frozenset({MediaKind.IMAGE.value}),
    MediaKind.VIDEO.value: frozenset({MediaKind.VIDEO.value}),
}


def build_bm25_body(
    query: str,
    *,
    modality_values: Collection[str],
    k: int,
    operator: str = "or",
    exclude_medical: bool = False,  # 2026-07-23 도메인 제외 전면 제거·기본 OFF(의료 이연). 복귀 시 True.
    search_filters: SearchFilters | None = None,
) -> dict[str, Any]:
    """BM25 필드별 named query 서브검색 본문(순수·결정적, 027 FR-001 · 044 FR-101).

    필드마다 별도 절(``bool.should`` + ``_name``)로 쪼개는 이유: 응답의 ``matched_queries`` 로
    **어느 필드에서 맞았는지**를 관측해야 뒤쪽 증거 판정(``query_evidence``)이 가능하기 때문이다.
    boost 차등(summary^3 … file_name^0.5)은 파일명 노이즈가 랭킹을 압도하지 못하게 한다.

    Args:
        query: 검색어(정규화가 끝난 문자열).
        modality_values: 이 버킷이 매칭할 저장 modality 값들. 정렬해 terms 필터로 넣는다.
        k: 가져올 문서 수(``size``).
        operator: ``or``(기본·한 토큰만 맞아도 후보) 또는 ``and``(**모든 토큰**이 맞아야 후보).
            ``and`` 는 정밀하지만 다어절 자연어 질의에서 결과가 비기 쉽다.
        exclude_medical: 의료 자산을 빼는 필터. **기본 꺼짐** — 도메인 균일 노출 결정에 따라
            현재 운영에서 켜지 않는다(의료 트랙 복귀 시 재사용할 자리).
        search_filters: 확장자·기간·주제 선필터. ``None`` 이면 modality 필터만 걸린다.

    Returns:
        OpenSearch 검색 본문 dict(순수 데이터 — 실행은 호출부가 한다).
    """
    filters: list[dict[str, Any]] = [{"terms": {"modality": sorted(modality_values)}}]
    filters.extend(filters_to_opensearch_bool(search_filters))
    label_term = query.strip().casefold()

    def _match(field: str, boost: float, _name: str) -> dict[str, Any]:
        """텍스트 필드 하나에 대한 match 절을 만든다(``_name`` 으로 hit 관측 가능하게)."""
        inner: dict[str, Any] = {"query": query, "_name": _name}
        if boost != 1.0:
            inner["boost"] = boost
        if operator != "or":
            inner["operator"] = operator
        return {"match": {field: inner}}

    def _cross_meta(*, boost: float, _name: str) -> dict[str, Any]:
        """summary+keywords 를 **한 필드처럼** 보는 절 — 토큰이 두 필드에 나뉘어 있어도 맞는다."""
        inner: dict[str, Any] = {
            "query": query,
            "type": "cross_fields",
            "fields": list(_CROSS_META_FIELDS),
            "_name": _name,
            "boost": boost,
        }
        if operator != "or":
            inner["operator"] = operator
        return {"multi_match": inner}

    should: list[dict[str, Any]] = [
        _match("keywords", 2.0, "hit_keywords"),
        {"term": {"labels": {"value": label_term, "_name": "hit_labels", "boost": 1.0}}},
        _match("file_name", 0.5, "hit_file_name"),
        _match("summary", 3.0, "hit_summary"),
        _cross_meta(boost=1.0, _name="hit_cross_meta"),
    ]
    # must_not = 의료 배제(exclude_medical·2026-07-23부터 기본 OFF). 활성일 때만 추가.
    must_not: list[dict[str, Any]] = []
    if exclude_medical:
        must_not.append({"term": {"domain_label": _MEDICAL_LABEL}})

    bool_clause: dict[str, Any] = {
        "should": should,
        "minimum_should_match": 1,
        "filter": filters,
    }
    if must_not:
        bool_clause["must_not"] = must_not
    return {"size": int(k), "query": {"bool": bool_clause}}


def build_knn_body(
    query_vector: list[float],
    *,
    modality_values: Collection[str],
    k: int,
    exclude_medical: bool = False,  # 2026-07-23 도메인 제외 전면 제거·기본 OFF(의료 이연). 복귀 시 True.
    search_filters: SearchFilters | None = None,
) -> dict[str, Any]:
    """plain kNN 단독 서브검색 본문(순수·결정적, 027 FR-001 — 게이트 신호용 kNN 표본 통합).

    정규화 파이프라인을 적용하지 않는 단일 knn 이다 — knn ``_score``(lucene cosinesimil=(1+cos)/2)에
    원시 코사인이 보존돼, 클라이언트가 ① 융합 기여(min-max)와 ② 게이트 신호(gate_signal)·per-result
    컷(_cos)을 **같은 표본 1회**로 얻는다(추가 게이트 검색 소멸 — SC-002). 023 게이트와 동일하게
    modality terms·의료배제를 **knn native filter(pre-filter)**로 적용한다 — bool 사후필터로 감싸면
    전역 k 최근접을 먼저 뽑고 걸러 작은 k 에서 비우세 모달리티(image 등)가 0 건이 되는 회귀(022 G3
    실OS 발견)를 막고, 그 모달리티 안에서 k 최근접을 뽑는다.

    Args:
        query_vector: 질의 임베딩(1536D).
        modality_values: 이 버킷이 매칭할 저장 modality 값들.
        k: 최근접 문서 수(``size`` 와 knn ``k`` 모두에 쓴다). 게이트 표본 하한 적용은 호출부 몫.
        exclude_medical: 의료 자산 제외. **기본 꺼짐**(도메인 균일 노출).
        search_filters: 확장자·기간·주제 선필터.

    Returns:
        OpenSearch knn 검색 본문 dict.
    """
    filters: list[dict[str, Any]] = [{"terms": {"modality": sorted(modality_values)}}]
    filters.extend(filters_to_opensearch_bool(search_filters))
    knn_filter: dict[str, Any] = {"bool": {"filter": filters}}
    must_not: list[dict[str, Any]] = []
    if exclude_medical:
        must_not.append({"term": {"domain_label": _MEDICAL_LABEL}})
    if must_not:
        knn_filter["bool"]["must_not"] = must_not
    return {
        "size": int(k),
        "query": {"knn": {"embedding": {"vector": list(query_vector), "k": int(k), "filter": knn_filter}}},
    }
