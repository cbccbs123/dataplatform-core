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
from collections.abc import Callable, Collection, Iterable
from typing import Any

from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind

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

# 게이트 기본값 — G4 실OS calibration 으로 확정(2026-06-10, dev 318자산·lucene cosinesimil probe).
# 검증 라벨셋(있음 9·없음 6)에서 probe 원시 코사인 top: 있음 최소 0.490 vs 없음 최대 0.365 → 갭 0.125,
# Δ(top-mean): 있음 최소 0.273 vs 없음 최대 0.179 → 갭 0.094 로 깨끗이 분리. FLOOR=0.43(top 갭 중앙·
# 양쪽 마진 ~0.06), EPS=0.15(없음 Δ는 차단하되 있음 약모달리티는 과필터 안 하도록 보수적·갭 하단).
# 절대 FLOOR 가 주분리축이고 상대 EPS 는 코퍼스 평탄 시 보강(AND). 임계는 probe 출력 스케일에 보정됨.
# ⚠️ 토픽 인접 없음(예: '양자컴퓨터'→과학 콘텐츠)은 코사인이 있음과 동률이라 게이트로 못 막는다 —
# nori 외래어 분해·임베딩 교정(Non-Goal·후속) 영역. 어휘적으로 무관한 없음(아이패드 등)은 확실히 차단.
_DEFAULT_CUTOFF_EPS = 0.15
_DEFAULT_CUTOFF_FLOOR = 0.43
_DEFAULT_PROBE_K = 50

# OS 검색 버킷 라벨 → 저장된 modality 값 집합(PG media_search 와 동일 분류). 요청 라벨('text')과
# 저장 modality 값('txt')의 불일치를 흡수한다 — text 버킷은 ALLOWED_TEXT_META_FILE_KINDS(txt·json·
# pdf·office), audio 는 'audio'. 단일 term 이 아니라 terms(집합) 필터를 써야 실데이터가 회수된다.
# 022 G1: image·video 추가. image/video 는 020 assets 인덱스에 한국어 VLM 캡션(nori) + KoSimCSE 캡션
# 임베딩(embedding)으로 이미 색인돼 있어 text/audio 와 **동일 OS 하이브리드**로 검색한다(CLIP 아님 —
# 시각-내용 매칭은 후속 spec). 저장값이 라벨과 동일('image'/'video')이라 search_assets_os 의 fallback
# frozenset({label}) 로도 동작하나, 지원 모달리티를 명시·문서화하려 여기 등재한다(행동은 동일).
_MODALITY_VALUES: dict[str, frozenset[str]] = {
    "text": frozenset(ALLOWED_TEXT_META_FILE_KINDS),
    "audio": frozenset({MediaKind.AUDIO.value}),
    MediaKind.IMAGE.value: frozenset({MediaKind.IMAGE.value}),
    MediaKind.VIDEO.value: frozenset({MediaKind.VIDEO.value}),
}


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
    modality_values: Collection[str],
    k: int = 100,
    weights: tuple[float, float] = (0.5, 0.5),
    exclude_medical: bool = True,
) -> dict[str, Any]:
    """OpenSearch 하이브리드 검색 본문을 만든다(순수·결정적).

    nori 텍스트 ``multi_match``(BM25) ⊕ ``knn``(embedding, 질의 벡터)의 **hybrid 쿼리**다. 두
    두 서브쿼리 모두 ① ``modality`` keyword **terms** 필터(``modality_values`` 집합 — text 버킷은 txt·
    json·pdf·office 등 다중값이라 terms 로 거른다)와 ② ``exclude_medical`` 이면 ``domain_label='medical'``
    ``must_not``(FR-011)을 적용하되, **적용 방식이 다르다**: 텍스트(BM25)는 ``bool`` 로 감싼 사후 필터로
    충분하나(BM25 는 k 제한이 없다), **knn 은 native ``filter``(pre-filter)**로 그 모달리티 안에서 k 최근접을
    뽑는다 — bool 사후필터는 전역 k 를 먼저 뽑고 걸러 작은 k 에서 비우세 모달리티가 비기 때문(G3 실OS 교정).

    점수 융합(BM25·kNN 의 min-max 정규화 + 가중평균)은 OpenSearch **검색 파이프라인**
    (normalization-processor, G2 ``ensure_search_pipeline``)이 담당하므로 본문은 hybrid 서브쿼리
    구성까지만 책임진다 — ``weights`` 는 그 파이프라인 등록용 융합 가중치 메타로, 쿼리 본문에는
    반영하지 않는다(호출부 search_assets_os 가 파이프라인에 전달, G2).

    ``size`` 는 ``k``. 필드명·차원은 020 인덱스 매핑(`opensearch_sync.build_index_body`)과 일치.

    **정렬**: OpenSearch hybrid 쿼리는 ``_score`` 와 필드 정렬 조합을 금지(실OS 400: "_score sort
    criteria cannot be applied with any other criteria")하므로 본문에 ``sort`` 를 두지 않는다 —
    정규화 융합 점수로 정렬되고, FR-009 결정적 tiebreaker(점수 desc·동점 ``id`` asc)는
    ``search_assets_os`` 가 클라이언트에서 적용한다(헌법 3조). (G5 실OS 검증으로 교정.)
    """
    # terms(집합) 필터 — 라벨↔저장값 불일치(text→txt·json…) 흡수. sorted 로 결정적 본문(헌법 3조).
    filters: list[dict[str, Any]] = [{"terms": {"modality": sorted(modality_values)}}]
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
    # knn 은 modality·의료배제를 **knn native filter**(pre-filter)로 적용한다 — bool 사후필터로 감싸면
    # 전역 k 최근접을 먼저 뽑고 거르므로, 작은 k(=버킷 한도)에서 비우세 모달리티(image 등)가 0 건이
    # 된다(G3 실OS 발견: image k=3 → 0건). filter 안에서 modality 로 좁혀 그 모달리티의 k 최근접을 뽑는다.
    knn_prefilter: dict[str, Any] = {"bool": {"filter": list(filters)}}
    if must_not:
        knn_prefilter["bool"]["must_not"] = list(must_not)
    knn_sub = {
        "knn": {
            "embedding": {"vector": list(query_vector), "k": int(k), "filter": knn_prefilter}
        }
    }

    return {
        "size": int(k),
        "query": {"hybrid": {"queries": [text_sub, knn_sub]}},
        # 정렬 없음: hybrid 쿼리는 _score+필드 sort 조합을 금지(실OS 400)하므로 정규화 융합 점수로만
        # 정렬한다. FR-009 결정적 tiebreaker(점수 desc·동점 id asc)는 search_assets_os 가 클라이언트에서.
    }


def build_probe_body(
    query_vector: list[float],
    *,
    modality_values: Collection[str],
    k: int,
    exclude_medical: bool = True,
) -> dict[str, Any]:
    """적합도 게이트용 plain kNN probe 본문(순수·결정적, 023 FR-002).

    정규화 검색 파이프라인을 **적용하지 않는** 단일 knn 이다 — 그 modality 의 원시 코사인(lucene
    cosinesimil _score)을 직접 얻어 게이트(passes_cutoff)에 쓴다. 하이브리드(BM25 융합)가 아니라
    knn 단독이라 _score 가 코사인 단조다. modality terms·의료배제(FR-008)를 knn native filter(pre-
    filter)로 적용해 **정규화 경로(build_search_body)와 같은 후보 집단**에서 신호를 잰다.
    """
    filters: list[dict[str, Any]] = [{"terms": {"modality": sorted(modality_values)}}]
    knn_filter: dict[str, Any] = {"bool": {"filter": filters}}
    if exclude_medical:
        knn_filter["bool"]["must_not"] = [{"term": {"domain_label": _MEDICAL_LABEL}}]
    return {
        "size": int(k),
        "query": {"knn": {"embedding": {"vector": list(query_vector), "k": int(k), "filter": knn_filter}}},
    }


def knn_score_to_cosine(score: Any) -> float:
    """lucene cosinesimil knn ``_score``(=(1+cos)/2)를 원시 코사인으로 환산한다(순수·결정적).

    020 인덱스 매핑이 ``space_type=cosinesimil``·``engine=lucene`` 이므로 lucene 의 코사인 knn 점수는
    ``(1+cos)/2`` 다 → ``cos = 2·score − 1``. 비유한·범위 밖은 [-1,1] 로 안전 clamp. 실 OS 의 정확한
    스케일·calibration 은 G4 실OS 에서 확정한다(여기선 환산식만 고정).

    비유한 _score 의 안전 기본값은 0.0(코사인 -1)이 아니라 **0.5(코사인 0, 중립)**다 — 환산식이
    ``score=0.5`` 에서 ``cos=0`` 이라, 무효 점수는 게이트를 끄는 쪽이 아니라 중립으로 떨어뜨려야 한다.
    """
    cos = 2.0 * _safe_float(score, 0.5) - 1.0
    return max(-1.0, min(1.0, cos))


# 게이트 경계 포용(>=) 부동소수 허용오차. 임계가 0.10·0.65 처럼 십진 근사라 0.75-0.65=0.0999…998
# 같은 표현오차로 **경계값이 탈락**하지 않게 한다(결정적·관측 무영향한 미소값).
_CUTOFF_TOL = 1e-9


def passes_cutoff(top: float, mean: float, *, eps: float, floor: float) -> bool:
    """적합도 게이트(순수·결정적, 023 FR-001·004): 그 modality 버킷을 유지할지 판정한다.

    ``keep = (top − mean) ≥ eps AND top ≥ floor``. 상대 신호 ``top − mean``(이방성 baseline 흡수)이
    주판정, 절대 ``floor`` 는 코퍼스 전체가 평평할 때의 느슨한 backstop. 둘 다 만족해야 유지(AND);
    실패면 빈 버킷(no-match → 무관 결과 표출 차단). 비교는 ``_CUTOFF_TOL`` 로 경계를 포용한다.
    """
    t = _safe_float(top)
    return (t - _safe_float(mean)) >= eps - _CUTOFF_TOL and t >= floor - _CUTOFF_TOL


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


def probe_relevance(
    client: Any,
    query_vector: list[float],
    *,
    modality_values: Collection[str],
    k: int,
    index: str,
    exclude_medical: bool = True,
) -> tuple[float, float]:
    """그 modality 의 원시 코사인 top·baseline(표본 평균)을 plain kNN probe 로 얻는다(023 FR-002).

    정규화 검색 파이프라인을 **적용하지 않고**(``search_pipeline`` 미지정) ``build_probe_body`` 로 knn
    단독 검색을 1회 한다 — knn ``_score`` 를 ``knn_score_to_cosine`` 로 원시 코사인 환산한 뒤 top(최대)·
    mean(표본 평균)을 돌려준다. hits 가 없으면 (0.0, 0.0)(게이트가 cut). OS 미도달은 그대로 전파(FR-007
    동형). 실 opensearch-py knn 응답·스케일은 G4 실OS 확정.
    """
    body = build_probe_body(query_vector, modality_values=modality_values, k=k, exclude_medical=exclude_medical)
    resp = client.search(index=index, body=body)
    hits = ((resp or {}).get("hits") or {}).get("hits") or []
    if not hits:
        return (0.0, 0.0)
    cosines = [knn_score_to_cosine(h.get("_score")) for h in hits]
    return (max(cosines), sum(cosines) / len(cosines))


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
    cutoff_enabled: bool = False,
    cutoff_eps: float = _DEFAULT_CUTOFF_EPS,
    cutoff_floor: float = _DEFAULT_CUTOFF_FLOOR,
    cutoff_probe_k: int = _DEFAULT_PROBE_K,
    probe_fn: Callable[..., tuple[float, float]] = probe_relevance,
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
    for label in modalities:
        # 요청 라벨('text'/'audio') → 저장된 modality 값 집합으로 해소(text=txt·json·pdf·office).
        # 매핑에 없는 라벨은 라벨 자체를 값으로 본다(미래 모달리티 안전 폴백).
        values = _MODALITY_VALUES.get(label, frozenset({label}))
        body = build_search_body(
            query,
            query_vector,
            modality_values=values,
            k=k,
            weights=weights,
            exclude_medical=exclude_medical,
        )
        resp = client.search(index=index, body=body, search_pipeline=pipeline_name)
        hits = ((resp or {}).get("hits") or {}).get("hits") or []
        rows = [os_hit_to_row(h) for h in hits]
        # FR-009 결정적 tiebreaker(헌법 3조): hybrid 쿼리는 OS sort(_score+필드)를 금지하므로 정렬을
        # 클라이언트에서 한다 — 정규화 융합 점수(similarity) desc, 동점은 id asc 로 결정적.
        rows.sort(key=lambda r: (-_safe_float(r.get("similarity")), str(r.get("id") or "")))
        if cutoff_enabled and rows:
            # 적합도 게이트(023): 정규화 랭킹은 불변, probe 신호로 버킷 유지/비움만 결정(FR-001·003).
            # 게이트 off(기본)면 probe 미호출·버킷 그대로 — 021/022 동작 보존(회귀 0). 이미 빈 버킷
            # (rows==[])은 probe 해도 결과가 빈 채이므로 단락해 불필요한 모달리티당 +1 kNN IO 를 아낀다.
            top, mean = probe_fn(
                client, query_vector, modality_values=values, k=cutoff_probe_k,
                index=index, exclude_medical=exclude_medical,
            )
            if not passes_cutoff(top, mean, eps=cutoff_eps, floor=cutoff_floor):
                rows = []  # no-match: 무관 결과 표출 대신 빈 버킷
        buckets[label] = rows
    return buckets


def get_client(url: str | None = None) -> Any:
    """검색용 OpenSearch 클라이언트 — 020 ``opensearch_sync.get_client`` 재사용(단일 출처)."""
    from src.search.opensearch_sync import get_client as _sync_get_client

    return _sync_get_client(url)
