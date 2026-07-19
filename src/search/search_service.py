"""하이브리드 검색 서비스 진입점 — 호출부(CLI/HTTP)에 독립적인 함수 계층.

요청(query·modalities·limit)을 받아 OpenSearch 하이브리드 검색으로 모달리티별 버킷 결과를 만든 뒤,
요청한 모달리티만 골라 일정한 모양으로 반환한다. 실제 검색·임베딩은 ``opensearch_search`` 가
담당하고, 본 모듈은 요청 정규화·채널 해소·응답 형태만 책임진다(F-4.3).

037 OpenSearch 전용 정리: 021 의 PG(``media_search`` FTS/벡터) 백엔드 분기를 걷어내고 OS 단일
경로만 남겼다. 내부에서는 ``_grouped_via_opensearch`` 만 호출한다. 069 US-C: 037 로 죽어 있던 PG 전용
no-op 인자(``structured``·``fusion``·``text_hybrid_alpha``·``image_search_alpha``·``chunk_agg``·
``min_scores``·``text_query_model``)와 하한 필터 잔재(``_filter_by_min_score``)를 시그니처·본문에서
철거했다 — ``backend`` 인자만 fail-fast(미지원 백엔드 ValueError) 가치로 남긴다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.config import search_constants
from src.config.embedding_constants import EMBEDDING_KIND_ST
from src.config.settings import (
    active_embed_channel,
    get_current_settings,
)

# 037: 검색 read path 단일 백엔드 = OpenSearch. opensearch_search 모듈 상단은 순수(opensearch-py·임베더는
# 함수 내부 지연 import)라 OS 미연결 환경(순수 단위 등)에서도 import 안전 — 실제 OS IO 는 search_hybrid
# 호출 시에만 발생한다.
from src.search.opensearch_search import get_client as os_get_client
from src.search.opensearch_search import normalize_query as os_normalize_query
from src.search.opensearch_search import search_assets_os as os_search_assets
from src.search.query_plan import build_query_plan, search_plan_to_meta
from src.search.search_filters import SearchFilters

_LOG = logging.getLogger(__name__)

# 요청 모달리티 라벨 → OS 버킷 결과 키.
# 결정성(헌법 3조): 결과 버킷 조립이 ``list(.items())`` 순회 순서에 의존하므로 삽입 순서를
# 보존한다(dict 는 3.7+ 삽입 순서 보장). set 등 순서 비보장 타입으로 대체 금지.
_MODALITY_BUCKETS: dict[str, str] = {
    "text": "text_documents",
    "audio": "audio",
    "image": "image",
    "video": "video",
}

# 022/037 백엔드 단일화: text·audio·**image·video 모두** 020 OS 인덱스(하이브리드)에서 검색한다(021 의
# image/video→PG CLIP 경로를 OS 로 전환·037 PG 경로 제거). image/video 는 020 assets 인덱스에 한국어 VLM
# 캡션(nori) + 활성 텍스트 채널 임베딩(embedding·현행 st_api·bge-m3)으로 이미 색인돼 있어 text/audio 와 동일 하이브리드로
# 회수된다(CLIP 아님 — 시각-내용 매칭은 후속 spec). 따라서 OS 경로는 요청 모달리티 전체를 한 번의
# search_assets_os 호출로 처리한다.

# 027: OS 융합·게이트·컷 기본값은 모듈 중복 상수(_DEFAULT_OS_*)를 두지 않고 src.config.search_constants
# 단일 출처(F1)를 직접 참조한다 — settings 미초기화(순수 단위 등) getattr 폴백도 같은 공개 상수를 쓴다
# (cross-module private import 없음·하드코딩 0). 미초기화 시 게이트 기본은 운영 기본과 동일(enabled True)이다.


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
    llm_verify_judge_fn: Callable[[str, str, str], bool] | None = None,
    search_mode: str = "auto",
    search_filters: SearchFilters | None = None,
    must_include: list[str] | None = None,
    must_exclude: list[str] | None = None,
) -> dict[str, Any]:
    """backend='opensearch' 경로의 모달리티 버킷을 조립한다(022·027, FR-002·FR-003·SC-005).

    text·audio·**image·video 모든 버킷**을 020 OS 인덱스에서 동일 하이브리드(nori BM25 캡션·라벨 +
    ``embedding`` kNN + **클라이언트 융합**)로 검색한다 — image/video 도 020 assets 인덱스에 한국어 VLM
    캡션·활성 텍스트 채널 임베딩(현행 st_api·bge-m3)으로 색인돼 있어 text/audio 와 같은 경로다(CLIP 아님; 시각-내용 매칭은 후속).
    요청 모달리티 전체를 **한 번의** ``os_search_fn`` 호출로 검색해 버킷을 만들고, 응답 표준 키
    (text_documents·audio·image·video·meta)로 담는다(응답 동형, SC-005).

    설계 판단:
    - **LLM 미접촉(FR-002·SC-004)**: 검색 질의 구조화 LLM 을 호출하지 않는다 — 멀티모달 LLM 0·ms.
      037 PG 검색 제거·069 US-C 정리로 PG 전용 파라미터(structured·alpha·fusion·query_model_name·
      chunk_agg 등)는 search_hybrid 시그니처에서 철거됐다(이 경로에서 쓰이지 않았음).
    - **query-norm 토글(072 — 029 seam 재사용, gemma 대체)**: ``search_os_query_norm_enabled`` on 이면
      검색 직전 **자연어 질의(어절≥3)**를 **nori 형태소 명사추출 + 모달리티어 스톱워드 제거**로 정규화
      (검색시점 LLM 0·결정적·``_analyze`` 사전 기반)한 뒤 OS 에 넘긴다 — 정규화를 service 레벨에서 1회만
      수행해 정규화된 질의가 OS seam 안에서 임베딩·BM25·rerank 채점에 동일 적용되게 한다. 단어 질의
      (어절<3)는 원문 그대로(정규화·_analyze 스킵·지연 0). 관측성은 top-level ``meta["query_norm"]`` 로
      노출(os_gate 미오염). off 면 원문 passthrough(바이트 동일·정규화 미호출, SC-002). ``query_norm_fn``
      은 테스트/커스텀 정규화 주입 seam(주입 시 형태소 대신 사용). **075**: ``search_os_query_norm_method``
      (``morph`` 기본 / ``llm``)로 방식을 고른다 — ``llm`` 이면 형태소 대신 gemma ``noun_phrase_query``
      (029 보존·검색시점 LLM)를 재배선하고 ``meta["query_norm"]["method"]`` 로 어느 방식이 돌았는지 노출한다.
    - **(buckets, gate_meta) 튜플 수신(027)**: ``os_search_fn``(search_assets_os)은 클라이언트 융합
      전환으로 버킷과 함께 게이트 메타(모달리티별 top·baseline·gate_passed·cut_count)를 돌려준다 →
      ``meta["os_gate"]`` 로 합류시켜 빈 버킷이 no-match 판정인지 즉시 관측 가능하게 한다(F4 관측성).
    - **모달리티 키 매핑**: ``os_search_fn`` 버킷 키는 모달리티명('text'/'image')이고 응답 grouped 키는
      ('text_documents'/'image')이므로 ``_MODALITY_BUCKETS`` 로 변환해 담는다.
    - **컷오프 설정(027)**: 게이트·per-result 컷 임계(eps·floor·result_floor·operator)를 cfg 에서 읽어
      OS seam 에 전달한다(getattr 폴백은 search_constants 단일 출처 — settings 미초기화 순수 단위 방어).
      ``disable_os_cutoff=True`` 면 ``cutoff_enabled=False`` 로 강제해 게이트·per-result 컷을 모두 끈다
      (no_cutoff 디버그 우회 — 약한 후보까지 노출). per-result 컷은 search_assets_os 내부 코사인 스케일
      (cut_rows·result_floor)에서 끝나므로 호출부에는 별도 하한 필터가 없다.
    - **OS 미도달(FR-007)**: ``os_client_fn``/``os_search_fn`` 예외를 try/except 로 감싸지 않아 그대로
      전파한다(silent pg 폴백 금지 — 결과가 백엔드 가용성에 따라 달라지지 않게).

    ⚠️ 결정성(헌법 3조): 최종 응답 버킷 순서는 호출부의 ``label_keys`` 가 정하므로 여기 grouped 의
    삽입 순서는 출력 순서에 영향하지 않는다.
    """
    requested = modalities if modalities is not None else list(_MODALITY_BUCKETS)

    if not requested:
        # 빈 요청: OS 미접촉(불필요 IO 회피). os_gate 도 빈 dict(검색 안 함).
        plan = build_query_plan(query, mode=search_mode)
        return {
            "meta": {
                "backend": "opensearch",
                "os_gate": {},
                "search_plan": search_plan_to_meta(plan),
            },
        }

    plan = build_query_plan(query, mode=search_mode)

    # disable_os_cutoff(디버그 우회)면 게이트·컷 모두 off. 아니면 cfg 의 enabled(미초기화 폴백=운영 기본).
    cutoff_enabled = (
        False
        if disable_os_cutoff
        else getattr(cfg, "search_os_cutoff_enabled", search_constants.OS_CUTOFF_ENABLED_DEFAULT)
    )

    # 072 query-norm(029 seam 재사용): cfg 토글(getattr 폴백=search_constants 단일 출처)을 읽어, on 이면
    # 검색 직전 질의를 **service 레벨에서 1회** 형태소 정규화한다. 정규화를 여기 한 곳에서 끝내는 이유:
    # ① 정규화 1회(중복 0), ② 관측성(FR-007)을 top-level meta["query_norm"] 로 노출해 모달리티 키 dict 인
    # os_gate(gate_meta)를 오염시키지 않음(골든 하니스 등 gate_meta 순회 소비자 보호 — search_assets_os 는
    # 별도 반환·gate_meta 오염 없이 정규화된 질의만 받음). off(기본)면 normalize_query 가 원문 그대로
    # 돌려줘 정규화 미호출·바이트 동일(SC-002). 정규화된 질의가 OS seam 안에서 임베딩·BM25·rerank 채점에
    # 동일 적용된다(query_norm_fn 미주입+enabled 면 아래 morph_noun_phrase_query 클로저를 배선).
    qn_enabled = getattr(
        cfg, "search_os_query_norm_enabled", search_constants.OS_QUERY_NORM_ENABLED_DEFAULT
    )
    # 072: 형태소 정규화가 nori _analyze(client)를 쓰므로 client 를 정규화 앞에서 획득한다(아래
    # os_search_fn 도 이 동일 client 를 재사용 — 생성 1회). OS 생성 실패 예외는 그대로 전파(FR-007).
    client = os_client_fn()
    # 075: 정규화 방식 선택(morph 기본·072 / llm·029 gemma). enabled on + query_norm_fn 미주입일 때만
    # 배선한다(주입 seam 우선). off 는 방식 무관 원문 passthrough(불변).
    qn_method = getattr(
        cfg, "search_os_query_norm_method", search_constants.OS_QUERY_NORM_METHOD_DEFAULT
    )
    norm_fn = query_norm_fn
    if qn_enabled and norm_fn is None:
        if qn_method == "llm":
            # 075: gemma(LLM) 정규화 경로(029 noun_phrase_query 재활성). 검색시점 LLM·단일 seam·temp=0·
            # fail-safe 원문 폴백. client=None → complete_json 이 현 설정 온프레미스 LLM 사용.
            from src.search.query_preprocess import noun_phrase_query

            def norm_fn(q: str) -> str:
                return noun_phrase_query(q)
        else:
            # 072(기본): 검색시점 질의 정규화를 **nori 형태소 명사추출+스톱워드**로(측정 2026-07-13: 자연어
            # nDCG 0.490→0.591 로 LLM 정규화 0.575 상회·검색시점 LLM 0·결정적). client·index 를 클로저로
            # 바인딩해 morph_noun_phrase_query 의 analyze_fn(nori _analyze IO)을 배선한다.
            from src.search.opensearch_search import nori_analyze_tokens
            from src.search.query_preprocess import morph_noun_phrase_query

            _norm_index = getattr(cfg, "opensearch_index", "assets")

            def norm_fn(q: str, _client: Any = client, _index: str = _norm_index) -> str:
                return morph_noun_phrase_query(
                    q,
                    analyze_fn=lambda text: nori_analyze_tokens(_client, text, index=_index),
                    stopwords=search_constants.OS_QUERY_NORM_STOPWORDS,
                    noun_pos=search_constants.OS_QUERY_NORM_NOUN_POS,
                    min_word_tokens=search_constants.OS_QUERY_NORM_MIN_WORD_TOKENS,
                )

    os_query = os_normalize_query(query, enabled=qn_enabled, llm_fn=norm_fn)

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
        # 073: aboutness OR-증거 필터(기본 off 상수 폴백 — settings 미초기화 순수 단위 방어).
        about_filter_enabled=getattr(
            cfg, "search_about_filter_enabled", search_constants.SEARCH_ABOUT_FILTER_ENABLED_DEFAULT
        ),
        search_mode=search_mode,
        search_policy=plan.policy,
        evidence_rescue_enabled=getattr(
            cfg, "search_evidence_rescue_enabled", search_constants.SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT
        ),
        evidence_debug=getattr(
            cfg, "search_evidence_debug", search_constants.SEARCH_EVIDENCE_DEBUG_DEFAULT
        ),
        search_filters=search_filters,
        # 057 FR-202: 서버 lexical 필터(must_include/exclude)를 OS seam 에 그대로 전달한다. None 은
        # 빈 리스트로 정규화해 넘겨 미지정과 동일한 하위호환 body(바이트 동일)를 보장한다.
        must_include=list(must_include or []),
        must_exclude=list(must_exclude or []),
    )  # client.msearch 미도달 예외도 전파(FR-007)

    # 074: 검색시점 top-3 개별 LLM 검증(L2) — 토글 on AND 자연어(어절≥3·072 판별과 동일 기준)일 때만.
    # 판정 질의=**사용자 원문**(의도 정보 보존·측정과 동일), 캐시 키=정규화 질의(표현 변형 흡수).
    # 단어 질의·off 는 검증 경로 무접촉(호출 0·응답 바이트 동일 — FR-001·005). 폴백은 모듈 내부(FR-003).
    lv_enabled = getattr(
        cfg, "search_llm_verify_enabled", search_constants.SEARCH_LLM_VERIFY_ENABLED_DEFAULT
    )
    llm_verify_meta: dict[str, Any] | None = None
    if lv_enabled and len((query or "").split()) >= search_constants.OS_QUERY_NORM_MIN_WORD_TOKENS:
        from src.search.llm_verify import verify_top_assets

        os_buckets, llm_verify_meta = verify_top_assets(
            os_buckets, query, norm_query=os_query, judge_fn=llm_verify_judge_fn
        )

    # meta 에 게이트 관측성(os_gate) + search_plan(044 FR-303) 합류.
    grouped: dict[str, Any] = {
        "meta": {
            "backend": "opensearch",
            "os_gate": gate_meta,
            "search_plan": search_plan_to_meta(plan),
        },
    }
    # 074 관측성(FR-006): 검증 실행 시에만 meta["llm_verify"] 노출(query_norm 관례 동형 — off 면 키 부재).
    if llm_verify_meta is not None:
        grouped["meta"]["llm_verify"] = llm_verify_meta
    # 029 query-norm 관측성(FR-007): on 일 때만 top-level meta["query_norm"] 로 원문→정규화 매핑을 노출
    # 한다(os_gate 는 모달리티 키 dict 이라 오염 금지). off(기본)면 키 자체를 두지 않아 027 meta 와 바이트
    # 동일(SC-001 — 기존 meta 형태 봉인 테스트 무영향).
    if qn_enabled:
        grouped["meta"]["query_norm"] = {
            "enabled": True, "method": qn_method, "original": query, "normalized": os_query,
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
    disable_os_cutoff: bool = False,
    text_channel: str | None = None,
    backend: str | None = None,
    _os_search_fn: Callable[
        ..., tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]
    ] = os_search_assets,
    _os_client_fn: Callable[..., Any] = os_get_client,
    _query_norm_fn: Callable[[str], str] | None = None,
    _llm_verify_judge_fn: Callable[[str, str, str], bool] | None = None,
    search_mode: str = "auto",
    search_filters: SearchFilters | None = None,
    must_include: list[str] | None = None,
    must_exclude: list[str] | None = None,
) -> dict[str, Any]:
    """질의를 OpenSearch 하이브리드 검색해 모달리티 버킷으로 반환한다.

    ``modalities`` 가 ``None`` 이면 전체 버킷(text/audio/image/video)을, 지정하면 해당
    버킷만 반환한다. 알 수 없는 모달리티 라벨은 ``ValueError``.

    037 OpenSearch 전용 정리: text·audio·image·video **모든 버킷**을 020 OS 인덱스(nori BM25 캡션·라벨
    + ``embedding`` kNN + 클라이언트 융합)에서 ``_os_search_fn`` 으로 검색해 같은 키(text_documents·
    audio·image·video)로 반환한다(022 — image/video 도 020 assets 인덱스에 한국어 VLM 캡션·활성 텍스트
    채널 임베딩(현행 st_api·bge-m3)으로 색인돼 text/audio 와 동일 하이브리드, CLIP 아님). OS 미도달이면
    ``_os_search_fn``/``_os_client_fn`` 예외를 **그대로 전파**한다(FR-007·SC-006 — silent 폴백 금지).

    ``disable_os_cutoff`` 는 OS 경로의 **게이트·per-result 컷을 모두 끄는 디버그 우회**다(기본 False).
    True 면 ``cutoff_enabled=False`` 로 전달돼 약한 후보까지 노출한다(포탈 ``/search?no_cutoff=true`` 배선·069 T407).
    **072 query-norm 토글**(``search_os_query_norm_enabled``, 029 seam 재사용)이 on 이면 검색 직전
    자연어 질의(어절≥3)를 **nori 형태소 명사추출+스톱워드 제거**(검색시점 LLM 0·결정적)한 뒤 임베딩·
    BM25·rerank 채점에 동일 적용한다 — 단어 질의(어절<3)·off 는 원문 ``query`` 그대로(바이트 동일).

    ``text_channel`` 은 텍스트 임베딩 채널 선택이다(텍스트 채널 한정). **미지정(None)** 이면 운영 활성
    프로파일(018, 적재·검색·관계 단일 출처)로 해소한다 — ``active_embed_channel()``. 017 A/B 하니스처럼
    **명시 전달은 그대로 우선**한다. 해소된 채널은 OS seam(``opensearch_search``)에 넘어가 질의 임베딩
    모델(``model_for_channel(channel)``)을 일치시킨다(FR-004 질의-문서 모델 일치). settings 미초기화
    (순수 단위 등)에서는 활성 해소가 ``RuntimeError`` 이므로 기존 기본 채널 ``'st'`` 로 보수적 폴백한다.

    ``_os_search_fn``/``_os_client_fn`` 은 테스트 주입 seam(기본 ``opensearch_search.search_assets_os``/
    ``get_client``). ``_query_norm_fn`` 은 query-norm seam(미주입+on 이면 072 ``morph_noun_phrase_query``
    형태소 클로저를 client·index 로 배선).

    **057 FR-202 서버 lexical 필터**: ``must_include``/``must_exclude`` 를 OS seam 에 그대로 넘겨 BM25
    본문의 must(전토큰 AND)/must_not 로 **전체 코퍼스**에 적용한다(프론트 페이지-only 필터의 서버 진실
    불일치 해소). 필터 절만 추가하고 융합·게이트·컷 로직은 무변경 → 랭킹 산식·정렬 불변. 미지정(None)이면
    빈 리스트로 넘어가 body 바이트 동일(하위호환·회귀 0).

    069 US-C(037 잔재 철거): 037 로 죽어 있던 PG 전용 no-op 인자(``structured``·``fusion``·
    ``text_hybrid_alpha``·``image_search_alpha``·``chunk_agg``·``min_scores``·``text_query_model``)와
    호출부 하한 필터(``_filter_by_min_score``)를 철거했다. per-result 컷은 ``search_assets_os`` 내부
    코사인 스케일(``cut_rows``·``result_floor``)에서 끝난다. ``backend`` 만 남긴다 — 'opensearch' 외
    값을 받으면 ``ValueError``(미지원 백엔드 fail-fast).
    """
    if modalities is not None:
        unknown = [m for m in modalities if m not in _MODALITY_BUCKETS]
        if unknown:
            raise ValueError(f"알 수 없는 모달리티: {unknown}")
        label_keys = [(m, _MODALITY_BUCKETS[m]) for m in modalities]
    else:
        label_keys = list(_MODALITY_BUCKETS.items())

    # 037: 백엔드는 OS 단일 경로다. backend 명시 인자는 하위호환 수용용이며 'opensearch' 외엔 미지원
    # (settings._SEARCH_BACKENDS 와 동형 fail-fast). 미지정(None)이면 OS 경로를 그대로 실행한다.
    if backend is not None and backend != "opensearch":
        raise ValueError(
            f"지원하지 않는 검색 백엔드: backend={backend!r} (지원: ['opensearch'])"
        )

    # 텍스트 임베딩 채널 해소(018, FR-004). 명시 전달은 그대로 우선(A/B). 미지정(None)이면 운영 활성
    # 프로파일(적재·검색·관계 단일 출처)로 해소한다 — text_channel 은 active_embed_channel().
    # OS 경로는 채널만 쓴다(질의 모델은 opensearch_search 가 model_for_channel(channel) 로 해소).
    # settings 미초기화(순수 단위 등)에서는 활성 해소가 RuntimeError 이므로 기존 기본 채널 'st' 로
    # 보수적 폴백한다(검색 단위가 settings 없이 그대로 동작).
    try:
        if text_channel is None:
            text_channel = active_embed_channel()
    except RuntimeError:
        # settings 미초기화: 운영 진입점(run_search 등)은 항상 init_settings 하므로 이 폴백은 비운영
        # (테스트) 경로다 — 오설정(운영서 init_settings 누락)을 관측 가능하게 warning 으로 남긴다(동작 불변).
        _LOG.warning("settings 미초기화 — 활성 임베딩 채널 'st' 보수 폴백(운영은 init_settings 필수)")
        if text_channel is None:
            text_channel = EMBEDDING_KIND_ST

    # OS 컷오프 설정을 읽기 위한 cfg. settings 미초기화(순수 단위 등)면 cfg=None → _grouped_via_opensearch
    # 의 getattr 폴백이 search_constants 단일 출처 기본값을 쓴다(F1).
    try:
        cfg = get_current_settings()
    except RuntimeError:
        cfg = None

    # 037: text·audio·image·video 전 모달리티를 OS(020 인덱스)로 검색한다. OS 미도달 예외는 전파(FR-007).
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
        llm_verify_judge_fn=_llm_verify_judge_fn,
        search_mode=search_mode,
        search_filters=search_filters,
        # 057 FR-202: 서버 lexical 필터를 OS 경로로 배선(랭킹 융합·컷오프 불변 — 필터 절만 추가).
        must_include=must_include,
        must_exclude=must_exclude,
    )
    # per-result 적합도 컷은 search_assets_os 내부 코사인 스케일(cut_rows·result_floor)에서 이미 끝나므로
    # 호출부에는 별도 하한 필터가 없다(069 US-C: 037 로 no-op 였던 _filter_by_min_score 철거). 요청 라벨
    # 키만 골라 표준 버킷으로 담는다.
    results = {key: grouped.get(key, []) for _label, key in label_keys}
    return {"query": query, "results": results, "meta": grouped.get("meta", {})}
