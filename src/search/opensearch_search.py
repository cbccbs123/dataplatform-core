"""OpenSearch 검색을 **실제로 실행**하는 계층 — 질의 임베딩 · 묶음 검색 · 형태소 분석.

**흐름에서의 위치**: 검색 본문은 ``query_builder`` 가 만들고, 여기서 보낸다. 돌아온 결과를
``fusion`` 이 합쳐 거르면 이 모듈이 버킷으로 담아 돌려준다. 인덱스는 **읽기만** 한다.

세 파일로 나뉘어 있지만 이 모듈이 나머지 둘의 함수를 **재export** 한다 — 그래서 호출부·테스트가
``opensearch_search.<이름>`` 으로 참조하거나 patch 하는 코드가 있고, 그게 그대로 동작한다.

핵심 동작: 모달리티마다 [벡터 kNN + BM25] 두 검색을 만들어 **한 번의 묶음 요청**으로 보낸다.
kNN 응답에 원시 코사인이 남아 있어, 추가 검색 없이 같은 표본으로 버킷 게이트와 결과 컷을 함께
판정할 수 있다.

opensearch-py 는 모듈 상단이 아니라 함수 안에서 import 한다 — 그 라이브러리가 없는 환경에서도
순수 함수만 쓰는 코드가 이 모듈을 import 할 수 있어야 하기 때문이다.

필드명 정본 = 020 인덱스 매핑(``opensearch_sync.build_index_body``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from src.config.search_constants import (
    OS_KNN_SAMPLE_K,
    OS_QUERY_NORM_DECOMPOUND,
    OS_QUERY_NORM_ENABLED_DEFAULT,
)
from src.search.bucket_policy import apply_bucket_policy

# US-E(FR-E5) 재export — 분리된 fusion(융합·게이트·컷 수학)·query_builder(본문 빌더) 심볼을
# 이 모듈 이름공간으로 끌어와 하위호환(import·monkeypatch 경로)을 보존한다. search_assets_os 가
# 아래 이름들을 **모듈 전역**으로 호출하므로 opensearch_search.<name> patch 가 그대로 적용된다.
from src.search.fusion import (
    cut_rows,
    fuse_hybrid,
    gate_signal,
    knn_score_to_cosine,
    minmax_normalize,
    normalize_query,
    os_hit_to_row,
    passes_cutoff,
    rerank_reorder,
)
from src.search.query_builder import (
    _MODALITY_VALUES,
    BM25_NAMED_QUERY_NAMES,
    build_bm25_body,
    build_knn_body,
)
from src.search.query_plan import SearchPolicy, build_search_policy
from src.search.search_filters import SearchFilters
from src.search.search_tuning import SearchTuning

# 재export 공개 표면(하위호환). __all__ 등재로 순수 재export 심볼의 F401 을 억제한다(의도적 re-export).
__all__ = [
    "BM25_NAMED_QUERY_NAMES",
    "_MODALITY_VALUES",
    "build_bm25_body",
    "build_knn_body",
    "cut_rows",
    "embed_query",
    "fuse_hybrid",
    "gate_signal",
    "get_client",
    "knn_score_to_cosine",
    "minmax_normalize",
    "nori_analyze_tokens",
    "normalize_query",
    "os_hit_to_row",
    "passes_cutoff",
    "rerank_reorder",
    "search_assets_os",
]

_LOG = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# IO 함수 (027) — opensearch-py·임베더는 모듈 상단이 아니라 **함수 내부에서 지연 import**.
# 플래그 off(pg 백엔드) 환경의 모듈 순수성(상단 import 없음)을 보존하기 위함이다 — 순수 함수만
# 쓰는 단위 게이트는 opensearch-py·임베더 미설치여도 import 가능해야 한다(020 동형).
# 실제 OS 동작 검증은 G4(실OS e2e). 여기 IO 는 가짜 msearch 클라이언트로 액션 조립을 단위 검증한다.
# ──────────────────────────────────────────────────────────────────────────


def embed_query(query: str, *, channel: str) -> list[float]:
    """질의를 문서와 **같은 채널·모델**로 임베딩한다.

    적재·검색이 공유하는 임베더를 그대로 쓴다 — 임베딩 로직을 복제하지 않는다. 채널에서 모델을
    찾는 규칙도 설정 한 곳에 있다. **인덱스에 든 벡터와 같은 모델로 만들어야** 비교가 성립한다.

    Args:
        query: 임베딩할 질의 텍스트.
        channel: 임베딩 채널. 모델·백엔드(로컬/원격 API) 해소의 단일 기준이다.

    Returns:
        1536D 질의 벡터.
    """
    from src.config.settings import model_for_channel
    from src.search.query_embed import embed_query_for_media_search

    # 062: channel 을 함께 넘겨 백엔드(로컬/API)까지 적재와 일치시킨다(st_api=API·그외 로컬).
    return embed_query_for_media_search(
        query, model_name=model_for_channel(channel), channel=channel
    )


def nori_analyze_tokens(
    client: Any, text: str, *, index: str, decompound: str = OS_QUERY_NORM_DECOMPOUND
) -> list[tuple[str, str]]:
    """OpenSearch 분석기로 텍스트를 ``(토큰, 품사)`` 목록으로 쪼갠다.

    질의 정규화가 쓰는 분석 함수의 실체다. 분해 모드를 요청마다 지정하므로 **재색인 없이 질의
    분석만** 바꿀 수 있고, 인덱스는 읽기만 한다.

    Args:
        client: OpenSearch 클라이언트.
        text: 분석할 텍스트. **비었으면 OS 를 호출하지 않고** 빈 리스트를 돌려준다.
        index: 분석기 설정을 빌려올 인덱스(읽기만 한다).
        decompound: nori 복합어 분해 모드. **재색인 없이 질의 분석만** 모드를 바꿀 수 있다.

    Returns:
        ``[(토큰, 품사코드)]``. 응답 키가 없으면 빈 리스트로 안전 처리한다.

    Raises:
        OpenSearch 미도달 예외를 그대로 올린다(조용한 폴백 금지).
    """
    if not text or not text.strip():
        return []
    body = {
        "tokenizer": {"type": "nori_tokenizer", "decompound_mode": decompound},
        "explain": True,
        "text": text,
    }
    resp = client.indices.analyze(index=index, body=body)  # OS 미도달 예외 전파(FR-007)
    tokenizer = ((resp or {}).get("detail") or {}).get("tokenizer") or {}
    out: list[tuple[str, str]] = []
    for t in tokenizer.get("tokens") or []:
        tok = t.get("token")
        if tok is None:
            continue
        pos = str(t.get("leftPOS") or "").split("(")[0]
        out.append((str(tok), pos))
    return out


def _resp_hits(resp: dict[str, Any] | None) -> tuple[list[dict[str, Any]], bool]:
    """묶음 검색 응답 하나에서 (hits, 오류 여부)를 안전하게 꺼낸다.

    묶음 검색은 전체가 성공(HTTP 200)이어도 **개별 검색만 실패**할 수 있다. 그것을 빈 결과로
    넘기면 "검색 결과 없음"과 구분되지 않으므로, 오류 여부를 함께 돌려 호출부가 표시하게 한다.

    Args:
        resp: 서브검색 응답 하나. ``None``(배열이 짧아 자리가 빈 경우)도 받는다.

    Returns:
        ``(hits, 오류 여부)``. 응답이 없거나 ``error`` 키가 있으면 ``([], True)``.
    """
    if not resp:
        return [], True  # 응답 누락(배열 짧음 등)도 오류로 본다
    if resp.get("error"):
        return [], True
    return (((resp.get("hits") or {}).get("hits")) or []), False


def search_assets_os(
    client: Any,
    query: str,
    *,
    modalities: Iterable[str],
    k: int = 20,
    channel: str = "st",
    index: str,
    exclude_medical: bool = False,  # 2026-07-23 도메인 제외 전면 제거·기본 OFF(의료 이연). 복귀 시 True.
    embed_fn: Callable[..., list[float]] = embed_query,
    tuning: SearchTuning = SearchTuning(),
    rerank_fn: Callable[..., list[float]] | None = None,
    query_norm_enabled: bool = OS_QUERY_NORM_ENABLED_DEFAULT,
    query_norm_fn: Callable[[str], str] | None = None,
    search_mode: str = "auto",
    search_policy: SearchPolicy | None = None,
    search_filters: SearchFilters | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """전 모달리티를 OpenSearch 에서 하이브리드 검색한다(HTTP **1회**·읽기 전용).

    한 번의 msearch 로 모달리티마다 [벡터 kNN + BM25] 두 서브검색을 보내고, 그 둘을 클라이언트에서
    융합한 뒤 버킷 게이트 → per-result 컷 순으로 걸러 담는다. kNN 표본을 요청 개수보다 크게 뽑는
    이유는 게이트 기준선(배경 수준)이 흔들리지 않을 만큼의 표본이 필요해서다.

    Args:
        client: OpenSearch 클라이언트. **읽기 전용으로만** 쓴다.
        query: 사용자 질의. 정규화가 켜져 있으면 이 함수 안에서 한 번 정규화되고, 그 결과가
            임베딩·BM25·리랭크에 **똑같이** 쓰인다.
        modalities: 검색할 버킷 라벨들(text·image·video·audio).
        k: 버킷당 응답 상한. kNN 표본은 게이트 안정성을 위해 이보다 크게 뽑을 수 있다.
        channel: 질의 임베딩 채널. **문서 색인과 같은 채널이어야** 같은 공간에서 비교된다.
        index: 검색할 인덱스 이름.
        exclude_medical: 의료 자산 제외. 기본 꺼짐(도메인 균일 노출).
        embed_fn: 질의 임베딩 함수 주입 seam(테스트가 네트워크 없이 대체).
        tuning: 게이트·컷·리랭크·증거 튜닝 묶음. 무인자 기본값 = 설정 상수 기본 동작.
        rerank_fn: 리랭커 채점 함수 주입. ``None`` 이면 필요할 때 기본 seam 을 지연 로드.
        query_norm_enabled: 질의 정규화 토글. 꺼짐(기본)이면 원문을 그대로 쓴다.
        query_norm_fn: 정규화 콜백 주입. 미주입인데 토글이 켜져 있으면 LLM 정규화를 지연 import
            한다 — 꺼진 환경에 LLM 의존을 끌어오지 않으려는 배치다.
        search_mode: ``auto``|``keyword``. ``search_policy`` 를 안 주면 이 값으로 정책을 만든다.
        search_policy: 이미 만들어 둔 정책. 주면 ``search_mode`` 보다 우선한다.
        search_filters: 확장자·기간·주제 선필터.

    Returns:
        ``(buckets, gate_meta)``. ``gate_meta[모달리티]`` 는 ``{top, baseline, gate_passed,
        cut_count}`` — 빈 버킷이 "정말 없음"인지 "게이트에 걸린 것"인지 구분하게 해 준다.

    Raises:
        OpenSearch 미도달 예외를 **그대로 올린다** — 조용히 빈 결과로 격하하면 장애가 no-match 와
        구분되지 않는다.
    """
    labels = list(modalities)
    # 029 query-norm(021 FR-004 토글 개정): 검색 직전 질의를 명사구로 **1회** 정규화한다(off=원문
    # passthrough·바이트 동일). 임베딩(embed_fn)·BM25(build_bm25_body)·rerank 채점이 모두 아래 단일
    # query 지역변수를 쓰므로 양쪽에 동일 적용된다. query_norm_fn 미주입이고 enabled 면 noun_phrase_query
    # 를 지연 import 한다 — LLM seam(complete_json)을 플래그 off 환경(순수 단위)에 당기지 않으려는 것(020
    # 동형). 단일 seam·temperature=0·env 입력 0(헌법 §3 결정성)은 noun_phrase_query 가 보장한다.
    norm_fn = query_norm_fn
    if query_norm_enabled and norm_fn is None:
        from src.search.query_preprocess import noun_phrase_query as norm_fn
    query = normalize_query(query, enabled=query_norm_enabled, llm_fn=norm_fn)
    policy = search_policy or build_search_policy(query, mode=search_mode)
    query_vector = embed_fn(query, channel=channel)
    sample_k = max(int(k), OS_KNN_SAMPLE_K)  # 게이트 표본 하한(robust baseline 안정용).

    # msearch 본문: 모달리티당 [헤더, knn, 헤더, bm25] 를 결정적 순서로 쌓는다(opensearch-py 규약 —
    # 각 서브검색 앞에 인덱스 헤더 1줄). 본문 순서가 결정적이라야 응답 분해도 결정적이다(헌법 3조).
    msearch_body: list[dict[str, Any]] = []
    for label in labels:
        # 요청 라벨('text'/'audio') → 저장된 modality 값 집합으로 해소(text=txt·json·pdf·office).
        # 매핑에 없는 라벨은 라벨 자체를 값으로 본다(미래 모달리티 안전 폴백 — 022 image/video 동형).
        values = _MODALITY_VALUES.get(label, frozenset({label}))
        knn_body = build_knn_body(
            query_vector, modality_values=values, k=sample_k, exclude_medical=exclude_medical,
            search_filters=search_filters,
        )
        bm25_body = build_bm25_body(
            query, modality_values=values, k=int(k),
            operator=tuning.bm25_operator, exclude_medical=exclude_medical,
            search_filters=search_filters,
        )
        msearch_body.extend(({"index": index}, knn_body, {"index": index}, bm25_body))

    resp = client.msearch(body=msearch_body)  # OS 미도달 예외 그대로 전파(FR-007)
    responses = (resp or {}).get("responses") or []

    buckets: dict[str, list[dict[str, Any]]] = {}
    gate_meta: dict[str, dict[str, Any]] = {}
    for i, label in enumerate(labels):
        knn_hits, knn_err = _resp_hits(responses[2 * i] if 2 * i < len(responses) else None)
        bm25_hits, bm25_err = _resp_hits(
            responses[2 * i + 1] if 2 * i + 1 < len(responses) else None
        )
        sub_error = knn_err or bm25_err
        if sub_error:
            # 부분 실패는 빈 버킷(no-match)과 구분돼야 한다 — meta 표식 + 경고 로그(FR-007).
            _LOG.warning("msearch 서브검색 부분 실패 modality=%s knn_err=%s bm25_err=%s", label, knn_err, bm25_err)

        # 융합 입력은 kNN **상위 k행만** — 게이트 표본(OS_KNN_SAMPLE_K)을 그대로 정규화에 쓰면
        # 분모가 넓어져 상위 경계(top-k) 순위가 흔들린다(서버 융합 시절의 결과셋 범위와 동치 유지).
        fused = fuse_hybrid(bm25_hits, knn_hits[: int(k)], weights=tuning.weights)
        # 게이트 신호는 kNN 원시 코사인 **전체 표본**에서 직접(probe 추가 호출 0 — robust baseline 용).
        knn_cosines = [knn_score_to_cosine(h.get("_score")) for h in knn_hits]
        top, baseline = gate_signal(knn_cosines)

        outcome = apply_bucket_policy(
            fused,
            query=query,
            top=top,
            baseline=baseline,
            k=int(k),
            tuning=tuning,
            rerank_fn=rerank_fn,
            policy=policy,
            passes_cutoff_fn=passes_cutoff,
            cut_rows_fn=cut_rows,
            rerank_reorder_fn=rerank_reorder,
        )
        buckets[label] = outcome.rows
        gate_meta[label] = {
            "top": top,
            "baseline": baseline,
            "gate_passed": outcome.gate_passed,
            "lexical_evidence": outcome.lexical_evidence,
            "cut_count": outcome.cut_count,
            "error": sub_error,
        }
        if outcome.rerank is not None:
            gate_meta[label]["rerank"] = outcome.rerank
    return buckets, gate_meta


def get_client(url: str | None = None) -> Any:
    """검색용 OpenSearch 클라이언트를 만든다 — 색인 쪽 팩토리를 그대로 재사용한다(단일 출처).

    Args:
        url: 접속 URL. ``None`` 이면 설정값을 쓴다.

    Returns:
        OpenSearch 클라이언트.
    """
    from src.search.opensearch_sync import get_client as _sync_get_client

    return _sync_get_client(url)
