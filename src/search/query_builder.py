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

# FR-011(헌법 10조 · 010 FR-014): 의료 자산은 검색 결과에서 제외(domain_label keyword 필터).
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


def _lexical_terms(terms: Collection[str] | None) -> list[str]:
    """057 FR-202: must_include/exclude 원시 텀을 정규화(strip·빈문자열 제거·순서 보존).

    빈/공백 텀은 빈 lexical 절을 만들지 않도록 걸러낸다(방어적) — 결과적으로 텀이 하나도 없으면
    호출부가 must/must_not 절을 아예 추가하지 않아 body 가 바이트 동일하게 유지된다(하위호환·회귀 0).
    순서는 요청 순서를 보존한다(같은 요청 → 같은 body·결정성, 헌법 3조).
    """
    return [s for s in ((t or "").strip() for t in (terms or ())) if s]


def _lexical_clause(term: str) -> dict[str, Any]:
    """057 FR-202: lexical 필터 텀 1개 → **cross_fields 전토큰(operator=and)** 매칭 절.

    ``must_include``(→ bool.must)·``must_exclude``(→ bool.must_not) 공통 형상이다. 대상 필드는
    기존 ``_CROSS_META_FIELDS``(summary^3·keywords^2)를 그대로 재사용해(단일 출처·표류 0) BM25 텍스트
    합본에 적용한다. 필터 목적이라 ``_name``(matched_queries 관측)은 붙이지 않는다 — should(스코어)
    절만 관측 대상이며, must/must_not 는 결과 집합을 좁히는 필터일 뿐(랭킹 산식 불변, FR-202).
    """
    return {
        "multi_match": {
            "query": term,
            "type": "cross_fields",
            "fields": list(_CROSS_META_FIELDS),
            "operator": "and",
        }
    }


def build_bm25_body(
    query: str,
    *,
    modality_values: Collection[str],
    k: int,
    operator: str = "or",
    exclude_medical: bool = False,  # 2026-07-23 도메인 제외 전면 제거·기본 OFF(의료 이연). 복귀 시 True.
    search_filters: SearchFilters | None = None,
    must_include: Collection[str] | None = None,
    must_exclude: Collection[str] | None = None,
) -> dict[str, Any]:
    """BM25 필드별 named query 서브검색 본문(순수·결정적, 027 FR-001 · 044 FR-101 · 057 FR-202).

    044: ``multi_match`` 대신 ``bool.should`` + ``_name`` 으로 ``matched_queries`` 관측.
    boost 차등(summary^3…file_name^0.5)은 clause ``boost`` 로 계승. operator='and' 면 match 절에
    전 토큰 매칭(025 FR-001). ``labels`` 는 keyword ``term``(+ casefold). ``size`` 는 ``k``.

    057 FR-202 서버 lexical 필터: ``must_include`` 각 텀은 ``bool.must`` 에(텀 간 AND — 전부 매칭돼야
    유지), ``must_exclude`` 각 텀은 ``bool.must_not`` 에(의료 배제 절과 공존) **cross_fields 전토큰**
    절로 삽입된다(``_lexical_clause``). 두 목록이 비면(미지정) ``must`` 키를 만들지 않고 ``must_not`` 도
    기존(의료 배제 or 없음) 그대로라 body 가 **바이트 동일**하다(하위호환·회귀 0). should(스코어)·filter·
    minimum_should_match 는 무변경 → 랭킹 융합·컷오프·정렬 불변(필터 절만 추가).
    """
    filters: list[dict[str, Any]] = [{"terms": {"modality": sorted(modality_values)}}]
    filters.extend(filters_to_opensearch_bool(search_filters))
    label_term = query.strip().casefold()

    def _match(field: str, boost: float, _name: str) -> dict[str, Any]:
        inner: dict[str, Any] = {"query": query, "_name": _name}
        if boost != 1.0:
            inner["boost"] = boost
        if operator != "or":
            inner["operator"] = operator
        return {"match": {field: inner}}

    def _cross_meta(*, boost: float, _name: str) -> dict[str, Any]:
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
    # 057 FR-202: lexical 필터 텀 정규화(빈/공백 제거). 둘 다 비면 아래에서 must 키 미생성·must_not
    # 기존 유지라 body 바이트 동일(하위호환).
    include_terms = _lexical_terms(must_include)
    exclude_terms = _lexical_terms(must_exclude)

    # must_not = 의료 배제(exclude_medical·2026-07-23부터 기본 OFF) + 057 must_exclude 텀 절. 둘 다
    # 없으면 키 자체를 두지 않는다. 의료 절이 활성일 때만 먼저 넣어 순서를 보존한다.
    must_not: list[dict[str, Any]] = []
    if exclude_medical:
        must_not.append({"term": {"domain_label": _MEDICAL_LABEL}})
    must_not.extend(_lexical_clause(t) for t in exclude_terms)

    bool_clause: dict[str, Any] = {
        "should": should,
        "minimum_should_match": 1,
        "filter": filters,
    }
    # must 키는 include 텀이 있을 때만 추가한다(미지정 시 키 부재 → 바이트 동일). must 절은 결과 집합을
    # 좁히는 lexical AND 필터이며 should(스코어) 절은 그대로라 랭킹 산식·컷오프·정렬은 불변(FR-202).
    if include_terms:
        bool_clause["must"] = [_lexical_clause(t) for t in include_terms]
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
    must_include: Collection[str] | None = None,
    must_exclude: Collection[str] | None = None,
) -> dict[str, Any]:
    """plain kNN 단독 서브검색 본문(순수·결정적, 027 FR-001 — 게이트 신호용 kNN 표본 통합).

    정규화 파이프라인을 적용하지 않는 단일 knn 이다 — knn ``_score``(lucene cosinesimil=(1+cos)/2)에
    원시 코사인이 보존돼, 클라이언트가 ① 융합 기여(min-max)와 ② 게이트 신호(gate_signal)·per-result
    컷(_cos)을 **같은 표본 1회**로 얻는다(추가 게이트 검색 소멸 — SC-002). 023 게이트와 동일하게
    modality terms·의료배제를 **knn native filter(pre-filter)**로 적용한다 — bool 사후필터로 감싸면
    전역 k 최근접을 먼저 뽑고 걸러 작은 k 에서 비우세 모달리티(image 등)가 0 건이 되는 회귀(022 G3
    실OS 발견)를 막고, 그 모달리티 안에서 k 최근접을 뽑는다. ``size`` 는 ``k``(게이트 표본 하한 적용은 호출부).
    """
    filters: list[dict[str, Any]] = [{"terms": {"modality": sorted(modality_values)}}]
    filters.extend(filters_to_opensearch_bool(search_filters))
    # 057 FR-202: must_include/exclude 를 kNN native pre-filter 에도 적용한다. BM25 서브검색만 필터하면
    # 융합(fuse union)에서 kNN 회수분이 필터를 우회해 실효가 없다(T213 골든 평가에서 발견). filter/
    # must_not(비스코어)로 넣어 kNN 후보를 must_include 매칭·must_exclude 비매칭 문서로 제한(전토큰 AND·
    # 랭킹 무영향). 빈/미지정이면 아래 절을 추가하지 않아 body 바이트 동일(하위호환).
    include_terms = _lexical_terms(must_include)
    exclude_terms = _lexical_terms(must_exclude)
    filters.extend(_lexical_clause(t) for t in include_terms)
    knn_filter: dict[str, Any] = {"bool": {"filter": filters}}
    must_not: list[dict[str, Any]] = []
    if exclude_medical:
        must_not.append({"term": {"domain_label": _MEDICAL_LABEL}})
    must_not.extend(_lexical_clause(t) for t in exclude_terms)
    if must_not:
        knn_filter["bool"]["must_not"] = must_not
    return {
        "size": int(k),
        "query": {"knn": {"embedding": {"vector": list(query_vector), "k": int(k), "filter": knn_filter}}},
    }
