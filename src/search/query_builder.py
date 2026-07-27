"""OpenSearch 검색 **본문(body) 빌더** — BM25·kNN 요청 dict 를 만드는 순수 함수들.

**흐름에서의 위치**: 여기서 만든 body 를 ``opensearch_search`` 가 실제로 보내고, 돌아온 결과는
``fusion`` 이 합쳐 거른다. 이 모듈은 OpenSearch 도 임베더도 건드리지 않아 단위 테스트로 전부 검증된다.

필드 이름의 정본은 색인 매핑(``opensearch_sync.build_index_body``)이다 — 텍스트(summary·keywords·
labels·file_name) · 벡터(embedding) · 필터용 keyword(modality·domain_label).

``opensearch_search`` 는 이 모듈의 함수들을 **재export** 한다 — 그래서 호출부·테스트가
``opensearch_search.build_bm25_body`` 로 참조하거나 patch 하는 코드가 있다(같은 함수다).
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from src.search.search_filters import SearchFilters, filters_to_opensearch_bool

# summary 와 keywords 를 **한 필드처럼** 묶어 보는 대상. 질의 토큰이 두 필드에 나뉘어 있어도
# 맞도록(예: "무선" 은 요약에, "충전기" 는 키워드에) 회수율을 올린다.
# ``^N`` 은 가중치다 — 요약·키워드를 파일명보다 크게 둬서 파일명 노이즈가 랭킹을 뒤집지 못하게 한다.
# 필드 이름의 정본은 색인 매핑(``opensearch_sync.build_index_body``)이다.
_CROSS_META_FIELDS: tuple[str, ...] = ("summary^3", "keywords^2")

# 각 BM25 절에 붙이는 이름. 응답의 matched_queries 로 **어느 필드에서 맞았는지** 관측해
# 증거 판정(query_evidence)에 쓰므로, 이 목록과 절 생성 코드는 1:1로 유지해야 한다.
BM25_NAMED_QUERY_NAMES: tuple[str, ...] = (
    "hit_keywords",
    "hit_labels",
    "hit_file_name",
    "hit_summary",
    "hit_cross_meta",
)

# 의료 자산을 걸러낼 때 쓰는 라벨. 현재 운영에서는 도메인을 균일하게 노출하므로 쓰이지 않는다.
_MEDICAL_LABEL = "medical"

# 요청 버킷 라벨 → 색인에 저장된 modality 값들.
# text 만 **집합**인 이유: 예전 문서에는 확장자 계열 값(txt·pdf 등)이 남아 있을 수 있어, 새 값과
# 옛 값을 함께 매칭해야 재색인 도중에도 결과가 비지 않는다. 그래서 term 이 아니라 terms 필터를 쓴다.
# 이미지·영상은 별도 벡터 공간이 아니라 텍스트와 같은 방식으로 찾으므로 값이 하나뿐이다.
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
    # 배제 절은 필요할 때만 넣는다 — 빈 must_not 을 항상 붙이면 요청 본문이 커지고,
    # 봉인 테스트가 보는 body 모양도 달라진다.
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
