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
from dataclasses import replace
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
from src.search.search_tuning import SearchTuning

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
) -> dict[str, Any]:
    """OpenSearch 경로의 모달리티 버킷을 조립한다.

    요청한 모달리티 **전체를 한 번의 검색 호출**로 처리하고, 결과를 응답 표준 키
    (text_documents·audio·image·video·meta)로 담는다. 질의 정규화·LLM 검증도 이 층에서 한 번씩만
    수행해, 정규화된 질의가 임베딩·BM25·리랭크에 똑같이 적용되게 한다.

    Args:
        query: 사용자 질의 원문.
        modalities: 검색할 모달리티 목록. ``None`` 이면 전체, **빈 리스트면 OpenSearch 를 아예
            호출하지 않고** 빈 결과를 돌려준다(불필요한 IO 회피).
        limit_per_bucket: 버킷당 결과 상한.
        text_channel: 텍스트 임베딩 채널(문서 색인과 일치해야 한다).
        cfg: 설정 객체. ``None`` 이면(순수 단위 테스트 등) 상수 기본값으로 폴백한다.
        disable_os_cutoff: 디버그 우회 — 게이트와 per-result 컷을 **모두 끈다**(약한 후보까지 노출).
        os_search_fn: 실제 검색 함수 주입 seam.
        os_client_fn: 클라이언트 팩토리 주입 seam. 여기서 만든 클라이언트를 정규화·검색이 공유한다.
        query_norm_fn: 정규화 콜백 주입. 주면 설정의 방식(형태소/LLM)보다 **우선**한다.
        llm_verify_judge_fn: 상위 결과 LLM 검증의 판정 함수 주입.
        search_mode: ``auto``|``keyword``.
        search_filters: 확장자·기간·주제 선필터.

    Returns:
        ``{text_documents, audio, image, video, meta}`` grouped dict. ``meta.os_gate`` 는 버킷별
        게이트 관측치(top·baseline·통과 여부·제거 수)라, **빈 버킷이 "정말 없음"인지 "게이트에
        걸린 것"인지** 바로 확인할 수 있다. 버킷 출력 순서는 호출부가 정한다.

    Raises:
        OpenSearch 미도달 예외는 감싸지 않고 그대로 올린다 — 결과가 백엔드 가용성에 따라
            달라지면 안 되기 때문이다.
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

    # 069 US-E(FR-E5②): OS 검색 튜닝 12종(융합·게이트·컷·rerank·evidence·about)을 cfg 에서 **1회**
    # 해소해 SearchTuning 한 묶음으로 만든다(종전 getattr 릴레이 12줄 제거·인자 축소). disable_os_cutoff
    # (디버그 우회)면 cutoff_enabled 만 False 로 덮는다(replace) — 게이트·per-result 컷 모두 off.
    # PR4b: settings 미초기화(순수 단위 등)로 cfg=None 이면 SearchTuning() 기본(search_constants·F1)으로
    # 폴백한다 — from_settings 는 완전한 cfg.search 를 요구하므로 None 가드를 호출부에 명시(방어 getattr 폐지).
    tuning = SearchTuning.from_settings(cfg) if cfg is not None else SearchTuning()
    if disable_os_cutoff:
        tuning = replace(tuning, cutoff_enabled=False)

    # 072 query-norm(029 seam 재사용): cfg 토글(getattr 폴백=search_constants 단일 출처)을 읽어, on 이면
    # 검색 직전 질의를 **service 레벨에서 1회** 형태소 정규화한다. 정규화를 여기 한 곳에서 끝내는 이유:
    # ① 정규화 1회(중복 0), ② 관측성(FR-007)을 top-level meta["query_norm"] 로 노출해 모달리티 키 dict 인
    # os_gate(gate_meta)를 오염시키지 않음(골든 하니스 등 gate_meta 순회 소비자 보호 — search_assets_os 는
    # 별도 반환·gate_meta 오염 없이 정규화된 질의만 받음). off(기본)면 normalize_query 가 원문 그대로
    # 돌려줘 정규화 미호출·바이트 동일(SC-002). 정규화된 질의가 OS seam 안에서 임베딩·BM25·rerank 채점에
    # 동일 적용된다(query_norm_fn 미주입+enabled 면 아래 morph_noun_phrase_query 클로저를 배선).
    # PR4b: cfg-파생 검색 설정을 여기서 한 번에 해소한다(방어 getattr 폐지·직접 중첩 접근). cfg=None
    # (settings 미초기화·순수 단위)이면 search_constants 단일 출처 기본으로 폴백(운영은 항상 init_settings).
    if cfg is not None:
        qn_enabled = cfg.search.os_query_norm_enabled
        qn_method = cfg.search.os_query_norm_method
        lv_enabled = cfg.search.llm_verify_enabled
        os_index_name = cfg.opensearch.index
    else:
        qn_enabled = search_constants.OS_QUERY_NORM_ENABLED_DEFAULT
        qn_method = search_constants.OS_QUERY_NORM_METHOD_DEFAULT
        lv_enabled = search_constants.SEARCH_LLM_VERIFY_ENABLED_DEFAULT
        os_index_name = "assets"
    # 072: 형태소 정규화가 nori _analyze(client)를 쓰므로 client 를 정규화 앞에서 획득한다(아래
    # os_search_fn 도 이 동일 client 를 재사용 — 생성 1회). OS 생성 실패 예외는 그대로 전파(FR-007).
    client = os_client_fn()
    # 075: 정규화 방식 선택(morph 기본·072 / llm·029 gemma). enabled on + query_norm_fn 미주입일 때만
    # 배선한다(주입 seam 우선). off 는 방식 무관 원문 passthrough(불변).
    norm_fn = query_norm_fn
    if qn_enabled and norm_fn is None:
        if qn_method == "llm":
            # 075: gemma(LLM) 정규화 경로(029 noun_phrase_query 재활성). 검색시점 LLM·단일 seam·temp=0·
            # fail-safe 원문 폴백. client=None → complete_json 이 현 설정 온프레미스 LLM 사용.
            from src.search.query_preprocess import noun_phrase_query

            def norm_fn(q: str) -> str:
                """LLM(gemma) 정규화 경로 — 실패 시 원문 폴백은 noun_phrase_query 가 보장한다."""
                return noun_phrase_query(q)
        else:
            # 072(기본): 검색시점 질의 정규화를 **nori 형태소 명사추출+스톱워드**로(측정 2026-07-13: 자연어
            # nDCG 0.490→0.591 로 LLM 정규화 0.575 상회·검색시점 LLM 0·결정적). client·index 를 클로저로
            # 바인딩해 morph_noun_phrase_query 의 analyze_fn(nori _analyze IO)을 배선한다.
            from src.search.opensearch_search import nori_analyze_tokens
            from src.search.query_preprocess import morph_noun_phrase_query

            _norm_index = os_index_name

            def norm_fn(q: str, _client: Any = client, _index: str = _norm_index) -> str:
                """형태소 정규화 경로 — client·index 를 기본값 인자로 **바인딩 시점에 고정**한다.

                클로저 변수로 잡지 않고 기본값으로 묶는 이유는, 이후 루프·재대입이 생겨도 이
                함수가 보는 값이 흔들리지 않게 하기 위해서다.
                """
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
        index=os_index_name,
        exclude_medical=False,  # 2026-07-23 도메인 제외 전면 제거(의료 이연). 복귀 시 True.
        # 069 US-E(FR-E5②): 융합·게이트·컷·rerank·evidence·about 튜닝 12종을 SearchTuning 한 묶음으로
        # 전달한다(위에서 from_settings 로 1회 해소·disable_os_cutoff 시 cutoff_enabled=False 덮음).
        # 종전 getattr 릴레이 12줄이 이 한 줄로 대체됐다(인자 축소·오타는 정적 검사로).
        tuning=tuning,
        search_mode=search_mode,
        search_policy=plan.policy,
        search_filters=search_filters,
    )  # client.msearch 미도달 예외도 전파(FR-007)

    # 074: 검색시점 top-3 개별 LLM 검증(L2) — 토글 on AND 자연어(어절≥3·072 판별과 동일 기준)일 때만.
    # 판정 질의=**사용자 원문**(의도 정보 보존·측정과 동일), 캐시 키=정규화 질의(표현 변형 흡수).
    # 단어 질의·off 는 검증 경로 무접촉(호출 0·응답 바이트 동일 — FR-001·005). 폴백은 모듈 내부(FR-003).
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
) -> dict[str, Any]:
    """질의를 OpenSearch 하이브리드 검색해 모달리티 버킷으로 반환한다.

    text·audio·image·video 를 **모두 같은 인덱스에서 같은 방식**(nori BM25 + 벡터 kNN + 융합)으로
    찾는다. 이미지·영상도 한국어 캡션과 텍스트 임베딩으로 색인돼 있어 별도 경로가 아니다.

    Args:
        query: 사용자 질의. 질의 정규화가 켜져 있으면 검색 직전 **한 번만** 정규화하고, 그 결과가
            임베딩·BM25·리랭크 채점에 똑같이 쓰인다(어절 3개 미만인 짧은 질의는 원문 그대로).
        modalities: 검색할 버킷. ``None`` 이면 전체.
        limit_per_bucket: 버킷당 결과 상한.
        disable_os_cutoff: 게이트·컷을 모두 끄는 디버그 우회(기본 꺼짐 — 약한 후보까지 노출된다).
        text_channel: 텍스트 임베딩 채널. **문서 색인과 같아야** 같은 벡터 공간에서 비교된다.
            ``None`` 이면 운영 활성 채널로 해소하고, 명시 전달은 그대로 우선한다(A/B 실험용).
            설정 미초기화 환경에서는 ``'st'`` 로 보수 폴백한다.
        backend: ``'opensearch'`` 만 지원한다.
        _os_search_fn: 검색 함수 주입 seam(테스트용).
        _os_client_fn: 클라이언트 팩토리 주입 seam(테스트용).
        _query_norm_fn: 질의 정규화 콜백 주입 seam. 미주입이고 토글이 켜져 있으면 설정된 방식
            (형태소 기본 / LLM)으로 배선한다.
        _llm_verify_judge_fn: 상위 결과 LLM 검증 판정 함수 주입 seam.
        search_mode: ``auto``|``keyword``.
        search_filters: 확장자·기간·주제 선필터.

    Returns:
        ``{text_documents, audio, image, video, meta}`` grouped dict.

    Raises:
        ValueError: 알 수 없는 모달리티 라벨이거나 ``backend`` 가 ``'opensearch'`` 가 아닐 때.
        OpenSearch 미도달 예외는 감싸지 않고 그대로 올린다 — 빈 결과로 감추면 장애가
            "검색 결과 없음"과 구분되지 않는다.

    설계 배경: ``specs/037-opensearch-only-cleanup`` · ``specs/072-search-morph-query-norm``
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

    # 여기서 정하는 건 **채널뿐**이다. 그 채널로 어떤 임베딩 모델을 쓸지는 검색 seam 이 다시
    # 해소하므로(모델명은 이 층에 없다), 채널만 맞으면 질의와 문서가 같은 공간에서 비교된다.
    try:
        if text_channel is None:
            text_channel = active_embed_channel()
    except RuntimeError:
        # 설정 미초기화는 테스트 경로뿐이다(운영 진입점은 항상 초기화한다) — 검색이 죽지 않게
        # 폴백하되, 운영에서 이 경고가 보이면 초기화 누락이므로 warning 으로 남긴다.
        _LOG.warning("settings 미초기화 — 활성 임베딩 채널 'st' 보수 폴백(운영은 init_settings 필수)")
        if text_channel is None:
            text_channel = EMBEDDING_KIND_ST

    # cfg=None 이면 아래에서 상수 기본값으로 폴백한다(설정 없이도 도는 단위 테스트 경로).
    try:
        cfg = get_current_settings()
    except RuntimeError:
        cfg = None

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
    )
    # 여기에 점수 하한 필터가 없는 것은 누락이 아니다 — 약한 결과 제거는 검색 seam 안에서
    # 코사인 척도로 이미 끝났다(같은 걸 여기서 또 걸면 이중 절삭이 된다).
    # 출력 순서는 이 dict 가 아니라 label_keys 순서가 정한다(요청 라벨만 골라 담는다).
    results = {key: grouped.get(key, []) for _label, key in label_keys}
    return {"query": query, "results": results, "meta": grouped.get("meta", {})}
