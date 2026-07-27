"""하이브리드 검색 서비스 진입점 — 호출부(CLI/HTTP)에 독립적인 함수 계층.

**흐름에서의 위치**: CLI·HTTP 가 이 모듈의 ``search_hybrid`` 를 부르면, 여기서 요청을 정리하고
(모달리티 검증·임베딩 채널 해소) ``opensearch_search`` 에 넘긴 뒤, 돌아온 버킷을 응답 모양으로 담는다.
검색·임베딩 자체는 하지 않는다 — **요청 정규화 · 채널 해소 · 응답 조립**이 이 층의 책임이다.

검색 백엔드는 OpenSearch 하나뿐이다. ``backend`` 인자가 남아 있는 것은 다른 값이 들어왔을 때
조용히 무시하지 않고 즉시 실패시키기 위해서다.
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

# 이 import 는 OS 연결이 없는 환경에서도 안전하다 — opensearch_search 모듈 상단은 순수하고,
# opensearch-py·임베더는 함수 안에서 지연 import 하기 때문이다(실제 통신은 검색 호출 때 처음 일어난다).
from src.search.opensearch_search import get_client as os_get_client
from src.search.opensearch_search import normalize_query as os_normalize_query
from src.search.opensearch_search import search_assets_os as os_search_assets
from src.search.query_plan import build_query_plan, search_plan_to_meta
from src.search.search_filters import SearchFilters
from src.search.search_tuning import SearchTuning

_LOG = logging.getLogger(__name__)

# 요청 모달리티 라벨 → 응답 버킷 키.
# ⚠️ **순서가 계약이다** — 응답 버킷 조립이 이 dict 의 순회 순서를 그대로 따르므로, set 처럼 순서를
# 보장하지 않는 타입으로 바꾸면 같은 질의가 매번 다른 순서를 내놓는다.
_MODALITY_BUCKETS: dict[str, str] = {
    "text": "text_documents",
    "audio": "audio",
    "image": "image",
    "video": "video",
}

# 이미지·영상도 텍스트와 **같은 인덱스·같은 방식**으로 찾는다 — 한국어 캡션과 텍스트 임베딩으로
# 색인돼 있기 때문이다(이미지 벡터로 직접 매칭하는 경로가 아니다). 그래서 모달리티가 몇 개든
# 검색 호출은 한 번이다.

# 게이트·컷 기본값은 이 모듈에 복사해 두지 않고 search_constants 를 직접 본다 — 값이 두 곳에 있으면
# 설정이 초기화된 경로와 안 된 경로가 서로 다르게 동작한다.


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

    # 튜닝값 12종을 여기서 **한 번에** 해소해 묶음으로 넘긴다 — 아래 호출부들이 설정을 각자 읽으면
    # 같은 요청 안에서 서로 다른 값을 볼 수 있다. cfg 가 없으면(설정 미초기화 테스트) 상수 기본값.
    tuning = SearchTuning.from_settings(cfg) if cfg is not None else SearchTuning()
    if disable_os_cutoff:
        # 디버그 우회는 게이트 스위치 하나만 덮는다 — 나머지 튜닝값은 운영과 동일하게 두어야
        # "컷만 없는 상태"를 관측할 수 있다.
        tuning = replace(tuning, cutoff_enabled=False)

    # 질의 정규화를 **검색층이 아니라 여기서** 하는 이유가 둘 있다.
    #   ① 한 번만 하면 되는 일이다 — 정규화된 질의를 아래로 넘기면 임베딩·BM25·리랭크가 모두
    #      같은 문자열을 쓴다(각 층이 따로 정규화하면 서로 다른 질의로 검색하게 된다).
    #   ② 관측치를 meta["query_norm"] 최상위에 실을 수 있다 — 검색층 반환값(모달리티별 dict)에
    #      끼워 넣으면 그 dict 를 순회하는 소비자들이 깨진다.
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
    # 클라이언트를 **정규화보다 먼저** 만든다 — 형태소 정규화가 OpenSearch 분석기를 호출하므로
    # 같은 클라이언트가 필요하고, 아래 검색도 이것을 재사용한다(연결 생성 1회).
    client = os_client_fn()
    # 정규화 방식은 설정이 고르지만, 콜백이 주입돼 있으면 그쪽이 우선이다(테스트·실험용 대체).
    norm_fn = query_norm_fn
    if qn_enabled and norm_fn is None:
        if qn_method == "llm":
            # LLM 방식은 검색 시점에 모델을 부른다 — 그래서 이 분기에서만 import 한다
            # (형태소 방식만 쓰는 환경에 LLM 의존을 끌어오지 않기 위해).
            from src.search.query_preprocess import noun_phrase_query

            def norm_fn(q: str) -> str:
                """LLM 정규화 경로 — 실패 시 원문 폴백은 ``noun_phrase_query`` 가 보장한다."""
                return noun_phrase_query(q)
        else:
            # 기본 경로. 형태소 분석기로 명사만 뽑으므로 검색 시점에 LLM 을 부르지 않고,
            # 사전 기반이라 같은 질의는 언제나 같은 결과가 된다.
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
        os_query,  # 정규화된 질의(꺼져 있으면 원문) — 임베딩·BM25·리랭크가 이 값을 함께 쓴다
        modalities=requested,  # 요청 모달리티를 한 번에 넘긴다(버킷마다 따로 호출하지 않는다)
        k=limit_per_bucket,
        channel=text_channel,
        index=os_index_name,
        exclude_medical=False,  # 도메인 균일 노출 — 의료 트랙을 다시 열 때 켤 자리
        tuning=tuning,
        search_mode=search_mode,
        search_policy=plan.policy,
        search_filters=search_filters,
    )  # OpenSearch 미도달 예외는 여기서 잡지 않고 호출자에게 그대로 올라간다

    # LLM 검증은 **자연어 질의에만** 건다 — 한두 단어 질의는 판정할 문맥이 없어 이득이 없고,
    # 검색 시점 모델 호출은 비싸기 때문이다(꺼져 있거나 짧은 질의면 이 경로 자체를 타지 않는다).
    # 판정에는 사용자 원문을, 캐시 키에는 정규화 질의를 쓴다(표현이 달라도 같은 판정을 재사용).
    llm_verify_meta: dict[str, Any] | None = None
    if lv_enabled and len((query or "").split()) >= search_constants.OS_QUERY_NORM_MIN_WORD_TOKENS:
        from src.search.llm_verify import verify_top_assets

        os_buckets, llm_verify_meta = verify_top_assets(
            os_buckets, query, norm_query=os_query, judge_fn=llm_verify_judge_fn
        )

    # meta 에 게이트 관측치와 질의 계획을 싣는다 — 빈 버킷의 원인을 응답만 보고 알 수 있게.
    grouped: dict[str, Any] = {
        "meta": {
            "backend": "opensearch",
            "os_gate": gate_meta,
            "search_plan": search_plan_to_meta(plan),
        },
    }
    # 검증이 실제로 돈 경우에만 키를 넣는다 — 키의 유무 자체가 "돌았는가"를 알려준다.
    if llm_verify_meta is not None:
        grouped["meta"]["llm_verify"] = llm_verify_meta
    # 정규화가 켜졌을 때만 원문→정규화 매핑을 노출한다 — 검색 결과가 이상할 때 "질의가 어떻게
    # 바뀌어 들어갔는지"를 응답만 보고 확인할 수 있어야 한다. 꺼져 있으면 키를 두지 않는다.
    if qn_enabled:
        grouped["meta"]["query_norm"] = {
            "enabled": True, "method": qn_method, "original": query, "normalized": os_query,
        }
    # 검색층은 모달리티 이름('text')으로 버킷을 주고, 응답 계약은 다른 키('text_documents')를 쓴다.
    # 그 이름 차이를 여기서 흡수한다.
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
