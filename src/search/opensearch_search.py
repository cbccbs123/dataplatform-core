"""OpenSearch 검색 쿼리 빌더·클라이언트 융합·결과 매핑 (검색 read path → OpenSearch, spec 021·027).

020 이 깐 단일 인덱스(nori 한국어 BM25 + ``knn_vector``)를 **읽기만** 한다(쓰기 0, 헌법 6조).
본 모듈의 **순수 함수**(서브검색 본문·융합·게이트·컷·결과 매핑)는 OS·DB·opensearch-py 없이
결정적으로 동작하며 단위 게이트에서 항상 검증된다(헌법 3조). 실제 검색 실행(IO: 질의 임베딩·
``search_assets_os`` msearch)의 opensearch-py 의존은 모듈 상단이 아니라 함수 내부에서 지연 import
한다(플래그 off 환경의 순수성 보존 — 020 동형).

**027 클라이언트 융합 재구성** — 왜 서버 파이프라인이 아니라 클라이언트인가:
종전(021~024)은 OS 서버 normalization-processor 로 BM25·kNN 을 융합해 응답에서 **원시 코사인이
소거**됐고, 그 보상으로 023 probe(같은 kNN 을 모달리티당 한 번 더 실행해 코사인 재구매)·024 정규화
스케일 임계 4종·빈 dict 센티넬이 쌓였다. 027 은 융합을 **클라이언트 순수 함수**로 옮긴다 — 모달리티당
[plain kNN + BM25] 서브검색을 전 모달리티 **_msearch 1회**로 받아 ``fuse_hybrid`` 가 min-max+가중평균
(서버와 동일 수식)으로 융합한다. kNN 응답에 원시 코사인이 자연히 보존돼 ① probe 가 개념째 소멸(중복
kNN 0), ② 게이트(``gate_signal``)·per-result 컷(``cut_rows``)이 **코사인 단일 스케일의 두 규칙**으로
통합, ③ 융합 수학이 서버 상태가 아닌 단위 검증 가능한 순수 함수로 이동(결정성). 임계 기본값은
``src.config.search_constants`` 단일 출처(F1).

필드명 정본 = 020 인덱스 매핑(`opensearch_sync.build_index_body`):
    - nori 텍스트: ``summary``·``keywords``·``labels``·``file_name``·``search_text``
    - 벡터: ``embedding``(knn_vector·1536D)
    - keyword 필터: ``modality``·``domain_label``
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Collection, Iterable
from typing import Any

from src.config.search_constants import (
    OS_BM25_OPERATOR_DEFAULT,
    OS_CUTOFF_ENABLED_DEFAULT,
    OS_CUTOFF_EPS_DEFAULT,
    OS_CUTOFF_FLOOR_DEFAULT,
    OS_FUSION_WEIGHTS_DEFAULT,
    OS_KNN_SAMPLE_K,
    OS_QUERY_NORM_ENABLED_DEFAULT,
    OS_RERANK_ENABLED_DEFAULT,
    OS_RERANK_MODEL_DEFAULT,
    OS_RERANK_TAU_DEFAULT,
    OS_RERANK_TOP_R_DEFAULT,
    OS_RESULT_FLOOR_DEFAULT,
    SEARCH_EVIDENCE_DEBUG_DEFAULT,
    SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT,
)
from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from src.search.query_evidence import evidence_score, lexical_rescue_keep, strong_evidence_score
from src.search.query_plan import SearchPolicy, build_search_policy

# 020 인덱스의 nori 텍스트 필드(BM25 multi_match 대상). 필드명 정본 = opensearch_sync.build_index_body.
# 주의: labels 는 매핑상 keyword 지만 plan §1 이 multi_match 대상에 포함한다 — multi_match 는 keyword
# 필드를 정확매칭 절로 안전 수용한다(텍스트 필드와 혼합 무해). text/audio 버킷 한국어 BM25 재현율용.
# 026 T008(FR-003①): boost 차등 표기('field^N') — 토픽 신호(summary^3·keywords^2·labels^1)를
# 파일명(file_name^0.5)보다 강하게 둬서 파일명 노이즈가 랭킹을 압도하지 못하게 한다(F8 구조 방어).
# search_text 는 boost 1(표기 생략) — 이제 file_name 을 합본하지 않으므로(026 T005) 안전하다.
# multi_match 는 'field^N' 부스트 문법을 그대로 해석한다.
_TEXT_FIELDS: tuple[str, ...] = (
    "summary^3",
    "keywords^2",
    "labels^1",
    "file_name^0.5",
    "search_text",
)

# 044 FR-101: named query _name 고정 집합(build_bm25_body bool.should).
BM25_NAMED_QUERY_NAMES: tuple[str, ...] = (
    "hit_keywords",
    "hit_labels",
    "hit_file_name",
    "hit_summary",
    "hit_search_text",
)

# FR-011(헌법 10조 · 010 FR-014): 의료 자산은 검색 결과에서 제외(domain_label keyword 필터).
_MEDICAL_LABEL = "medical"

# 027 T006: 게이트 기본 상수(중복 cutoff eps·floor·표본 수)는 모듈 중복 출처라 제거했다 — 임계
# 기본값의 단일 출처는 src.config.search_constants(F1). 값은 코사인 스케일로 통일됐다(클라이언트
# 융합 전환으로 서버 출력 스케일 보정이 불필요해짐, 027).

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


def build_bm25_body(
    query: str,
    *,
    modality_values: Collection[str],
    k: int,
    operator: str = "or",
    exclude_medical: bool = True,
) -> dict[str, Any]:
    """BM25 필드별 named query 서브검색 본문(순수·결정적, 027 FR-001 · 044 FR-101).

    044: ``multi_match`` 대신 ``bool.should`` + ``_name`` 으로 ``matched_queries`` 관측.
    boost 차등(summary^3…file_name^0.5)은 clause ``boost`` 로 계승. operator='and' 면 match 절에
    전 토큰 매칭(025 FR-001). ``labels`` 는 keyword ``term``(+ casefold). ``size`` 는 ``k``.
    """
    filters: list[dict[str, Any]] = [{"terms": {"modality": sorted(modality_values)}}]
    label_term = query.strip().casefold()

    def _match(field: str, boost: float, _name: str) -> dict[str, Any]:
        inner: dict[str, Any] = {"query": query, "_name": _name}
        if boost != 1.0:
            inner["boost"] = boost
        if operator != "or":
            inner["operator"] = operator
        return {"match": {field: inner}}

    should: list[dict[str, Any]] = [
        _match("keywords", 2.0, "hit_keywords"),
        {"term": {"labels": {"value": label_term, "_name": "hit_labels", "boost": 1.0}}},
        _match("file_name", 0.5, "hit_file_name"),
        _match("summary", 3.0, "hit_summary"),
        _match("search_text", 1.0, "hit_search_text"),
    ]
    bool_clause: dict[str, Any] = {
        "should": should,
        "minimum_should_match": 1,
        "filter": filters,
    }
    if exclude_medical:
        bool_clause["must_not"] = [{"term": {"domain_label": _MEDICAL_LABEL}}]
    return {"size": int(k), "query": {"bool": bool_clause}}


def build_knn_body(
    query_vector: list[float],
    *,
    modality_values: Collection[str],
    k: int,
    exclude_medical: bool = True,
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


def passes_cutoff(top: float, baseline: float, *, eps: float, floor: float) -> bool:
    """적합도 게이트(순수·결정적, 023 FR-001·004 계승): 그 modality 버킷을 유지할지 판정한다.

    ``keep = (top − baseline) ≥ eps AND top ≥ floor``. 상대 신호 ``top − baseline``(background 흡수)이
    주판정, 절대 ``floor`` 는 코퍼스 전체가 평평할 때의 느슨한 backstop. 둘 다 만족해야 유지(AND);
    실패면 빈 버킷(no-match → 무관 결과 표출 차단). 비교는 ``_CUTOFF_TOL`` 로 경계를 포용한다.

    027: ``baseline`` 은 ``gate_signal`` 이 주는 **하위 절반 평균**(robust)이다 — 023 의 전체 평균을
    대체해 밀집 토픽('충전')에서 baseline 이 끌려 올라가는 오컷을 구조 해결한다. 입력 의미만 바뀌고
    판정식·경계 처리는 동일(시그니처 (top, baseline, *, eps, floor) — 인자 위치·의미 보존).
    """
    t = _safe_float(top)
    return (t - _safe_float(baseline)) >= eps - _CUTOFF_TOL and t >= floor - _CUTOFF_TOL


def minmax_normalize(scores: Iterable[float]) -> list[float]:
    """점수 리스트를 [0,1] 로 min-max 정규화한다(순수·결정적, 027 FR-002).

    클라이언트 융합의 1단계다 — OS 서버 normalization-processor 가 하던 min-max 를 **순수 함수로
    이관**(헌법 3조: 융합 수학을 서버 상태가 아닌 단위 검증 가능한 코드로). 빈 입력은 ``[]``,
    퇴화(``max==min``, 단일 원소·전 동점)는 **전원 1.0** 으로 결정적 정의한다(0 나눗셈 회피·랭킹
    보존). 입력 순서·길이를 보존해 호출부(fuse_hybrid)가 hit 리스트와 자리표시로 zip 할 수 있다.
    """
    vals = [_safe_float(s) for s in scores]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0.0:
        # max==min(단일·전 동점): 정규화 불가 → 전원 1.0(서버 min_max 의 퇴화 처리와 동형·결정적).
        return [1.0 for _ in vals]
    return [(v - lo) / span for v in vals]


def gate_signal(cosines: Iterable[float]) -> tuple[float, float]:
    """kNN 원시 코사인 리스트에서 게이트 신호 ``(top, robust baseline)`` 을 뽑는다(순수·결정적, FR-003).

    023 probe(추가 kNN 호출)를 대체한다 — 융합이 클라이언트로 오면서 kNN 응답에 원시 코사인이
    자연히 보존되므로, **추가 검색 없이** 같은 kNN 표본에서 신호를 잰다(SC-002: probe 소멸).

    - ``top`` = 표본 최댓값(그 modality 의 최적합 신호).
    - ``baseline`` = **코사인 하위 절반 평균**(robust). 023 의 '전체 평균' 은 밀집 토픽(예: '충전'
      → 다수 영상이 고코사인)에서 baseline 이 끌어올려져 ``top−baseline`` 이 좁아지는 오컷을 낳았다.
      하위 절반(정렬 후 floor(n/2)개)만 평균내면 background(무관 꼬리) 해석이 유지돼 밀집 토픽에서도
      신호가 분리된다 — 신규 분기·상수 0 의 **구조 해결**(027 핵심).

    표본이 2개 미만이면 하위 절반을 평균낼 수 없어 ``baseline=0.0``(빈 표본은 ``top`` 도 0.0).
    """
    vals = [_safe_float(c) for c in cosines]
    if not vals:
        return (0.0, 0.0)
    top = max(vals)
    if len(vals) < 2:
        return (top, 0.0)
    ordered = sorted(vals)
    half = len(ordered) // 2  # 하위 절반 개수(floor) — 정렬 하위 구간만 background 로.
    lower = ordered[:half]
    baseline = sum(lower) / len(lower)
    return (top, baseline)


def cut_rows(rows: Iterable[dict[str, Any]], *, result_floor: float) -> list[dict[str, Any]]:
    """per-result 컷(순수·결정적, FR-004): 행 유지 = ``_bm25`` 매칭 **OR** ``_cos ≥ result_floor``.

    024 의 정규화 스케일 모달리티별 임계 4종을 **코사인 스케일 단일 임계**로 통일한다.
    - ``_bm25 is True`` (전 토큰 어휘 증거)면 유지 — 코사인이 없어도(``_cos None``, BM25-only 행) 안전.
    - 아니면 원시 코사인 ``_cos`` 가 ``result_floor`` 이상일 때만 유지(의미 증거). ``_cos None`` 은
      비교 불가이므로 ``_bm25`` 가 유일 근거다(없으면 컷).
    입력(이미 융합 정렬된) 순서를 보존한다 — 랭킹 불변, 노이즈 꼬리만 제거. 경계는 ``_CUTOFF_TOL`` 포용.
    """
    kept: list[dict[str, Any]] = []
    for r in rows:
        if r.get("_bm25"):
            kept.append(r)
            continue
        cos = r.get("_cos")
        if cos is not None and _safe_float(cos) >= result_floor - _CUTOFF_TOL:
            kept.append(r)
    return kept


def rerank_reorder(
    rows: Iterable[dict[str, Any]],
    query: str,
    rerank_fn: Callable[..., list[float]] | None = None,
    *,
    top_r: int,
    tau: float = 0.0,
    model_name: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    """게이트·컷 통과 행을 cross-encoder 로 **재정렬**한다(순수·결정적, 029 FR-002 — G3 실측 정정).

    029 augment 의 잎 함수다 — 게이트(``passes_cutoff``)가 '없음' 버킷을 비우고 ``cut_rows`` 가
    행을 선별(027·recall·차단 보존)한 **뒤**, 통과 버킷의 생존자를 재정렬한다.
    ⚠️ **cut_rows 를 대체하지 않는다**. 초기 029 구현은 리랭크가 선별까지(τ 드롭·R 절삭) 맡았으나
    G3 실OS 측정에서 recall 0.9396→0.8999·차단 23→22 로 회귀했다(τ 가 약한-정답 행을 드롭하고
    R=10 절삭이 융합 11~20위 정답을 잘랐다). 028 의 +1.7%p 하이브리드는 '선별=cut_rows·리랭크=
    재정렬'이었다 — 리랭크는 **순서만** 바꾼다.

    핵심(전역 union 랭킹과의 정합): 호출부의 자산 단위 합집합 랭킹이 ``similarity`` 로 재정렬하므로,
    리랭크 점수(0~1 절대)를 그대로 덮으면 융합 점수(상대·min-max)와 **스케일이 어긋나** 모달리티
    간 순위가 깨진다(reranked 행의 작은 절대점수가 타 버킷 융합점수에 밀려 recall 손실). 그래서
    상위 R 행의 **융합 similarity 값 집합은 보존**하되 리랭크 점수 순서대로 **재배정**한다 —
    best-reranked 행이 그 head 의 최고 융합점수를 받는다. 결과: 버킷이 union 에 기여하는 점수
    분포 불변(recall 보존) + head 내 행 순서만 리랭크(p@3↑).

    - ``head = rows[:top_r]`` 만 채점(지연 통제)·``tail = rows[top_r:]`` 은 융합 순서 유지(뒤에 붙임).
      문서측 입력은 ``_rrtext``("요약: …\n키워드: …") 우선·없으면 ``summary`` 폴백(028 구조화 입력).
    - ``tau > 0`` 이면 head 중 rerank 점수 < τ 행을 **선택적** 드롭(정밀 필터). 기본 ``τ=0`` 은
      드롭 0·순수 재정렬 — G3 실측상 τ 드롭이 recall 손실이라 augment 기본은 재정렬만이다.
    - 빈 head 는 ``(list(rows), [])``(rerank_fn 미호출). ``rerank_fn is None`` 이면 기본 seam
      ``src.search.reranker.score_pairs`` 를 지연 import(무거운 의존을 플래그 off 환경에 안 당김).

    반환 ``(reordered_rows, scores)`` — ``scores`` 는 head 채점값(관측 ``rerank.top``·``scored`` 용).
    """
    rows = list(rows)
    head = rows[: int(top_r)]
    tail = rows[int(top_r):]
    if not head:
        return rows, []
    if rerank_fn is None:
        from src.search.reranker import score_pairs as rerank_fn  # noqa: PLW2901 — lazy 기본 seam
    scores = rerank_fn(
        query,
        [str(r.get("_rrtext") or r.get("summary") or "") for r in head],
        model_name=model_name,
    )
    pairs = list(zip(head, scores, strict=True))
    if tau > 0.0:
        pairs = [(r, sc) for r, sc in pairs if float(sc) >= tau]
    if not pairs:
        return tail, list(scores)
    # 리랭크 순서(점수 desc·동점 id asc)로 head 정렬.
    pairs.sort(key=lambda p: (-_safe_float(p[1]), str(p[0].get("id") or "")))
    # 융합 similarity 값 집합 보존 → 리랭크 순서에 내림차순 재배정(best-reranked = 최고 융합점수).
    # union 기여 similarity 다중집합 불변 → recall 근사 보존(G3 실측 0.9370·−0.0026 경계 타이) +
    # 순서만 리랭크(p@3 +2.2%p). reranked 행과 tail 은 자연히 분리(head 가
    # 융합 상위라 tail 보다 큰 점수 — 재배정 후에도 head ≥ tail).
    fusion_sims = sorted((_safe_float(r.get("similarity")) for r, _ in pairs), reverse=True)
    reordered = [{**r, "similarity": fusion_sims[i]} for i, (r, _sc) in enumerate(pairs)]
    return reordered + tail, list(scores)


def normalize_query(
    query: str,
    *,
    enabled: bool,
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """검색 질의를 LLM 핵심 명사구로 정규화하는 순수 토글 함수(029 FR-004, 기본 off).

    028 후속 측정이 확인한 명사구 정규화('별 보는 방법'→'천체 관측' 0.009→0.227, 패러프레이즈
    오컷 직격)를 **설정 토글**로 들인다. 021 FR-004(검색시점 LLM 금지)를 거버넌스 절차로 개정한
    뒤에만 켜며(G2), 본 함수는 그 토글의 **순수 잎**이다 — ``llm_fn`` 주입 seam·네트워크 0·결정적.

    - ``enabled=False``(기본): 원문 그대로 반환한다 — **바이트 동일 passthrough**(027 회귀 0의 봉인
      지점). 빈/``None`` 질의도 정규화할 내용이 없으므로 그대로 반환한다(llm_fn 미호출 안전).
    - ``enabled=True``: ``llm_fn(query)`` 가 돌려준 명사구를 반환한다. ``llm_fn`` 의 결정성(temp=0·
      env 입력 0)은 호출부(G2 ``query_norm`` seam)가 보장한다 — 본 함수는 같은 입력에 같은 출력.
      방어적으로 ``llm_fn`` 미주입(``None``)이면 정규화 수단이 없으므로 원문을 그대로 둔다(배선
      누락이 결정성을 깨지 않게 — fail-safe to 027).
    """
    if not enabled or not query or llm_fn is None:
        return query
    return llm_fn(query)


def fuse_hybrid(
    bm25_hits: Iterable[dict[str, Any]],
    knn_hits: Iterable[dict[str, Any]],
    *,
    weights: tuple[float, float],
) -> list[dict[str, Any]]:
    """BM25·kNN 두 서브검색 hit 을 클라이언트에서 융합한다(순수·결정적, 027 FR-002).

    서버 normalization-processor(min-max + arithmetic_mean)를 **순수 함수로 이관**한다(헌법 3조 —
    융합 수학이 서버 상태가 아니라 단위 검증 가능한 코드로). 수식·가중치는 동일:

      similarity = w_bm25 · norm(BM25 _score) + w_knn · norm(kNN 코사인)

    - 두 측을 각각 ``minmax_normalize`` 한다(BM25 는 원시 _score, kNN 은 ``knn_score_to_cosine`` 환산
      코사인). 한쪽에만 있는 자산은 **누락측 정규화 기여 0**(서버 hybrid 와 동일 시맨틱 — plan R2).
    - 자산 id 로 **합집합**한다. 같은 자산이 양쪽에 있으면 **한 행**으로 결합(점수만 합산). 행의 메타
      (file_uri·modality·summary)는 먼저 본 측(BM25 우선) hit 으로 채운다 — 같은 asset_id 라 내용 동일.
    - 행은 ``os_hit_to_row`` 동형 + 내부키 ``_cos``(kNN 원시 코사인|없으면 None)·``_bm25``(BM25 매칭 여부).
      이 두 키가 게이트(gate_signal)·per-result 컷(cut_rows)의 **단일 코사인 스케일 신호**다(024 통일).
    - 정렬은 ``(-similarity, id)`` — 점수 desc·동점 id asc 결정적(FR-002, 헌법 3조).
    """
    w_bm25, w_knn = weights
    bm25_list = list(bm25_hits)
    knn_list = list(knn_hits)
    bm25_norm = minmax_normalize([_safe_float(h.get("_score")) for h in bm25_list])
    knn_cos = [knn_score_to_cosine(h.get("_score")) for h in knn_list]
    knn_norm = minmax_normalize(knn_cos)

    # id → {row, norm_bm25, norm_knn, cos, bm25}. 누락측은 0(기여 0)·_cos None·_bm25 False 가 기본.
    merged: dict[str, dict[str, Any]] = {}

    def _entry(hit: dict[str, Any]) -> dict[str, Any]:
        row = os_hit_to_row(hit)
        # 028: rerank 문서측 입력은 **구조화 텍스트**("요약: …\n키워드: …") — 입력 변형 4종
        # 실측에서 최선(회식 0.003→0.071·인접-없음 차단 유지). 요약문에 없는 토픽 앵커(예:
        # 회식의 keywords '술자리')를 자연어 골격 안에 제공한다. nori 토큰 나열은 패러프레이즈
        # 붕괴(별보기 0.005 — 문맥 파괴)로 기각. 내부키 — 응답 전 제거.
        src_ = hit.get("_source") or {}
        _kw = src_.get("keywords") if isinstance(src_.get("keywords"), list) else []
        _summ = str(src_.get("summary") or row.get("summary") or "")
        row["_rrtext"] = (
            f"요약: {_summ}\n키워드: {' '.join(str(x) for x in _kw)}" if _kw else _summ
        )
        aid = row["id"]
        e = merged.get(aid)
        if e is None:
            # 먼저 본 측의 메타로 행을 채운다(BM25 루프가 먼저라 양쪽 자산은 BM25 hit 메타 — 내용 동일).
            e = merged.setdefault(
                aid, {"row": row, "norm_bm25": 0.0, "norm_knn": 0.0, "cos": None, "bm25": False}
            )
        return e

    # minmax_normalize 가 입력 길이를 보존하므로 zip 양변 길이가 같다(strict=True 로 불변 보장).
    for hit, norm in zip(bm25_list, bm25_norm, strict=True):
        e = _entry(hit)
        e["norm_bm25"] = norm
        e["bm25"] = True
    for hit, norm, cos in zip(knn_list, knn_norm, knn_cos, strict=True):
        e = _entry(hit)
        e["norm_knn"] = norm
        e["cos"] = cos

    out: list[dict[str, Any]] = []
    for e in merged.values():
        row = dict(e["row"])
        row["similarity"] = w_bm25 * e["norm_bm25"] + w_knn * e["norm_knn"]
        row["_cos"] = e["cos"]
        row["_bm25"] = e["bm25"]
        out.append(row)
    out.sort(key=lambda r: (-_safe_float(r.get("similarity")), str(r.get("id") or "")))
    return out


def os_hit_to_row(hit: dict[str, Any]) -> dict[str, Any]:
    """OpenSearch hit 을 media_search 버킷 행과 동형(SC-005)인 dict 로 매핑한다(순수·결정적).

    media_search 버킷 행의 핵심 키(``id``·``file_uri``·``modality``·``summary``·``similarity``·
    ``domain_label``)에 맞춘다 — 검색 서비스가 OS 버킷과 PG 버킷을 같은 모양으로 병합할 수 있게
    한다(응답 동형). ``domain_label`` 은 020 인덱스 ``_source`` 에서 그대로 옮겨 042 포탈 tier
    projection·``group_ranked`` 의료 배제 2차 방어에 쓴다(사용자 검색 파라미터 아님).
    ``similarity`` 는 hit 의 원시 ``_score`` 다(서브검색 단독 점수) — 027 클라이언트 융합에서는
    ``fuse_hybrid`` 가 이 값을 **융합 점수로 덮어쓴다**(min-max+가중평균). ``_source`` 누락·메타
    None 도 안전 처리한다.

    ⚠️ media_search 버킷 행의 자산 식별 키는 ``asset_id`` 가 아니라 ``id`` 다(검색 SQL alias
    ``asset_id AS id``). 따라서 020 인덱스 ``_source.asset_id``(== ``_id``)를 이 ``id`` 로 옮긴다.
    """
    src = hit.get("_source") or {}
    asset_id = src.get("asset_id") or hit.get("_id")
    row: dict[str, Any] = {
        "id": str(asset_id) if asset_id is not None else None,
        "file_uri": str(src.get("fs_uri") or ""),
        "modality": src.get("modality"),
        "domain_label": src.get("domain_label"),
        "summary": str(src.get("summary") or ""),
        "similarity": _safe_float(hit.get("_score"), 0.0),
    }
    mq = hit.get("matched_queries")
    if mq is not None:
        row["matched_queries"] = [str(x) for x in mq]
    return row


# ──────────────────────────────────────────────────────────────────────────
# IO 함수 (027) — opensearch-py·임베더는 모듈 상단이 아니라 **함수 내부에서 지연 import**.
# 플래그 off(pg 백엔드) 환경의 모듈 순수성(상단 import 없음)을 보존하기 위함이다 — 위 순수 함수만
# 쓰는 단위 게이트는 opensearch-py·임베더 미설치여도 import 가능해야 한다(020 동형).
# 실제 OS 동작 검증은 G4(실OS e2e). 여기 IO 는 가짜 msearch 클라이언트로 액션 조립을 단위 검증한다.
# 027 T006: 서버 융합 파이프라인 등록 코어와 별도 게이트 검색(추가 kNN) 경로는 제거됐다 — 융합이
# 클라이언트로 이동해 서버 상태(파이프라인) 0·중복 kNN 0(SC-002).
# ──────────────────────────────────────────────────────────────────────────


def embed_query(query: str, *, channel: str) -> list[float]:
    """질의를 인덱스와 **동일 채널·모델**로 임베딩한다(FR-004 질의-문서 일치, 018 단일 출처).

    적재·검색이 공유하는 텍스트 임베더(`query_embed.embed_query_for_media_search`)를 **재사용**한다
    — 임베딩 로직을 복제하지 않는다. 채널→모델 해소는 단일 출처 ``settings.model_for_channel``
    (017 A/B)로 한다(기본 ``'st'`` → KoSimCSE, ``'st_bge'`` → BGE-M3). 020 인덱스가 활성 채널
    임베딩을 색인하므로, 질의도 같은 채널 모델로 임베딩해야 같은 벡터 공간에서 비교된다.
    """
    from src.config.settings import model_for_channel
    from src.search.query_embed import embed_query_for_media_search

    return embed_query_for_media_search(query, model_name=model_for_channel(channel))


_LOG = logging.getLogger(__name__)


def _resp_hits(resp: dict[str, Any] | None) -> tuple[list[dict[str, Any]], bool]:
    """msearch 한 서브검색 응답에서 (hits, 오류여부)를 안전 추출한다(None·누락 방어).

    msearch 는 HTTP 200 이어도 서브검색별로 ``{"error": …}`` 를 돌려줄 수 있다 — 이를 조용히
    빈 결과로 격하하면 **부분 실패가 no-match 와 구분 불가**해진다(FR-007 관측성·silent 폴백
    금지 취지). 오류 여부를 함께 돌려 호출부가 gate_meta 에 표식·로그를 남긴다(리뷰 후속).
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
    weights: tuple[float, float] = OS_FUSION_WEIGHTS_DEFAULT,
    index: str,
    exclude_medical: bool = True,
    embed_fn: Callable[..., list[float]] = embed_query,
    cutoff_enabled: bool = OS_CUTOFF_ENABLED_DEFAULT,
    cutoff_eps: float = OS_CUTOFF_EPS_DEFAULT,
    cutoff_floor: float = OS_CUTOFF_FLOOR_DEFAULT,
    result_floor: float = OS_RESULT_FLOOR_DEFAULT,
    bm25_operator: str = OS_BM25_OPERATOR_DEFAULT,
    rerank_enabled: bool = OS_RERANK_ENABLED_DEFAULT,
    rerank_top_r: int = OS_RERANK_TOP_R_DEFAULT,
    rerank_tau: float = OS_RERANK_TAU_DEFAULT,
    rerank_model: str = OS_RERANK_MODEL_DEFAULT,
    rerank_fn: Callable[..., list[float]] | None = None,
    query_norm_enabled: bool = OS_QUERY_NORM_ENABLED_DEFAULT,
    query_norm_fn: Callable[[str], str] | None = None,
    search_mode: str = "auto",
    search_policy: SearchPolicy | None = None,
    evidence_rescue_enabled: bool = SEARCH_EVIDENCE_RESCUE_ENABLED_DEFAULT,
    evidence_debug: bool = SEARCH_EVIDENCE_DEBUG_DEFAULT,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """전 모달리티를 020 OS 인덱스에서 **클라이언트 융합** 검색한다(027 FR-001·002·003·004·007).

    흐름(검색 1회당 OS HTTP **1회**):
    (a) ``embed_fn`` 으로 질의를 **1회만** 임베딩해 벡터를 전 modality 재사용한다(중복 임베딩 0).
    (b) modality 마다 [plain kNN(k=max(요청k, ``OS_KNN_SAMPLE_K``)) + BM25(k=요청k)] 두 서브검색
        본문을 만들어 **_msearch 1회**로 보낸다. msearch 본문 순서는 ``[m1-knn, m1-bm25, m2-knn,
        m2-bm25, …]`` 로 결정적(헌법 3조) — 응답을 같은 순서로 분해한다. kNN 을 표본 하한까지 키우는
        이유는 robust baseline(하위 절반 평균)이 흔들리지 않을 표본을 그 modality 안에서 확보하기 위함.
    (c) ``fuse_hybrid`` 가 두 서브검색을 합집합·min-max+가중평균 융합한다(서버 파이프라인과 동일 수식,
        순수 함수). 행에 원시 코사인(_cos)·BM25 매칭(_bm25)이 동반된다.
    (d) **버킷 게이트**: kNN 코사인 표본에서 ``gate_signal`` → (top, robust baseline) → ``passes_cutoff``.
        실패면 그 버킷을 비운다(no-match → 무관 결과 표출 차단, FR-003).
    (e) **per-result 컷**: 게이트 통과 버킷에 ``cut_rows``(BM25 매칭 OR cos≥``result_floor``)로 노이즈
        꼬리를 제거한다(FR-004, 단일 코사인 스케일).
    (f) 내부키(_cos·_bm25)를 제거하고 요청 k 로 상한해 버킷에 담는다(SC-005 응답 동형·구 size=k 계약 보존).

    반환 ``(buckets, gate_meta)`` — ``gate_meta[modality] = {top, baseline, gate_passed, cut_count}``
    (F4 관측성: 빈 버킷이 no-match 판정인지 즉시 확인). ``cut_count`` 는 융합 행 중 게이트·컷으로
    제거된 수(상한 절삭은 포함 안 함 — 컷 효과만).

    **디버그 우회**(``cutoff_enabled=False``): 게이트·per-result 컷을 **모두 끈다** — 융합 전체를 그대로
    노출한다(약한 후보까지 관측). 호출부(search_service)의 ``disable_os_cutoff`` 가 이 스위치로 배선된다.

    OS 미도달(``client.msearch`` 예외)이면 **그대로 전파**한다(FR-007 — silent pg 폴백 금지: 결과가
    백엔드 가용성에 따라 달라지면 결정성·관측성 훼손). 검색은 OS 를 **읽기 전용**으로만 만진다(헌법 6조).

    **검색시점 LLM 질의 정규화(021 FR-004 — 029 토글 개정)**: 기본 off(``query_norm_enabled=False``)면
    원문 질의를 nori·임베딩에 직접 쓴다(021 현행·회귀 0). ``query_norm_enabled=True`` 면 검색 직전 질의를
    ``normalize_query`` 로 **핵심 명사구 정규화**(gemma·temp=0·env 입력 0·src/llm/client 단일 seam)한 뒤
    임베딩·BM25·rerank 채점에 **동일 적용**한다 — 021 이 LLM-free 로 만든 OS 읽기 경로를 무단 회귀시키는
    것이 아니라, 헌법 §Governance 절차로 021 FR-004 를 기본 off 토글 허용으로 정식 개정한 결과다(§3 결정성
    제약 강제). ``query_norm_fn`` 미주입이고 enabled 면 ``noun_phrase_query`` 를 지연 import 해 쓴다.
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
    label_values: list[frozenset[str]] = []
    for label in labels:
        # 요청 라벨('text'/'audio') → 저장된 modality 값 집합으로 해소(text=txt·json·pdf·office).
        # 매핑에 없는 라벨은 라벨 자체를 값으로 본다(미래 모달리티 안전 폴백 — 022 image/video 동형).
        values = _MODALITY_VALUES.get(label, frozenset({label}))
        label_values.append(values)
        knn_body = build_knn_body(
            query_vector, modality_values=values, k=sample_k, exclude_medical=exclude_medical
        )
        bm25_body = build_bm25_body(
            query, modality_values=values, k=int(k),
            operator=bm25_operator, exclude_medical=exclude_medical,
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
        fused = fuse_hybrid(bm25_hits, knn_hits[: int(k)], weights=weights)
        # 게이트 신호는 kNN 원시 코사인 **전체 표본**에서 직접(probe 추가 호출 0 — robust baseline 용).
        knn_cosines = [knn_score_to_cosine(h.get("_score")) for h in knn_hits]
        top, baseline = gate_signal(knn_cosines)

        has_lexical = any(r.get("_bm25") for r in fused)
        # 029 augment(replace→augment): rerank 는 게이트를 **대체하지 않고** 그 위에 얹는다. 우선순위를
        # 코드로 강제한다 — ① 디버그 우회(cutoff off → 게이트·컷·rerank 모두 off) → ② 버킷 게이트
        # (passes_cutoff)가 '없음' 버킷을 먼저 비움(rerank 가 못 하는 분포 기반 거부 — 차단 23/24 보존)
        # → ③ **통과 버킷 안에서만** rerank 가 cut_rows 를 대체(τ 선별·점수 재정렬). rerank 가 결과를
        # 가르는 잎은 'cutoff_enabled ∧ gate_passed' 단 하나다 — 게이트 실패·어휘 구제·빈 버킷 잎은 불변.
        rerank_info: dict[str, Any] | None = None  # gate_passed ∧ rerank 잎에서만 meta 에 부착(FR-007)
        if not cutoff_enabled:
            # 디버그 우회(FR-003): 게이트·per-result 컷·rerank 를 **모두 off** → 융합 전체 노출(약한
            # 후보까지 관측). 우회를 게이트·rerank 보다 **먼저** 둬, 종전 replace 경로가 rerank 를 cutoff
            # 검사보다 앞세워 숨겼던 모호성을 코드로 해소한다.
            kept = fused
            gate_passed = True
            cut_count = 0
        else:
            gate_passed = passes_cutoff(top, baseline, eps=cutoff_eps, floor=cutoff_floor)
            if gate_passed:
                # 선별 = 027 cut_rows(BM25 매칭 OR cos≥result_floor) — recall·no-match 차단 보존.
                kept = cut_rows(fused, result_floor=result_floor)
                cut_count = len(fused) - len(kept)
                if rerank_enabled and kept:
                    # augment(028 1순위·G3 정정): 리랭크는 cut_rows 생존자를 **재정렬만** 한다(대체 아님).
                    # 융합 상위 R건을 (질의, 구조화 텍스트) 쌍으로 채점 → head 순서를 리랭크로 바꾸되
                    # 융합 점수 분포는 보존(rerank_reorder). 기본 τ=0 은 드롭 0(재정렬만).
                    kept, scores = rerank_reorder(
                        kept, query, rerank_fn,
                        top_r=rerank_top_r, tau=rerank_tau, model_name=rerank_model,
                    )
                    cut_count = len(fused) - len(kept)  # τ>0 선택적 드롭 반영(기본 τ=0 은 추가 컷 0).
                    rerank_info = {
                        "enabled": True, "scored": len(scores), "kept": len(kept),
                        "top": (max(scores) if scores else 0.0),
                    }
            elif has_lexical and bm25_operator == "and":
                # 044 evidence rescue: env off → 027 legacy(어휘 BM25 행 전부 keep). env on →
                # matched_queries·policy 로 행 단위 keep/drop(FR-202). gate_passed 경로는 불변.
                kept = []
                for row in fused:
                    if not row.get("_bm25"):
                        continue
                    keep, reason = lexical_rescue_keep(
                        row.get("matched_queries"),
                        policy=policy,
                        rescue_enabled=evidence_rescue_enabled,
                    )
                    if keep:
                        tagged = dict(row)
                        tagged["_keep_reason"] = reason
                        kept.append(tagged)
                cut_count = len(fused) - len(kept)
            else:
                kept = []  # 어휘 증거도 없음 → 빈 버킷(no-match)
                cut_count = len(fused) - len(kept)

        # 내부키(_cos·_bm25·_rrtext·_keep_reason) 제거 + debug meta(FR-405) + 요청 k 상한.
        clean: list[dict[str, Any]] = []
        for row in kept[: int(k)]:
            out_row = {
                key: val
                for key, val in row.items()
                if key not in ("_cos", "_bm25", "_rrtext", "_keep_reason")
            }
            if evidence_debug:
                mq = out_row.get("matched_queries")
                out_row["evidence_score"] = evidence_score(mq)
                out_row["strong_evidence_score"] = strong_evidence_score(mq)
                out_row["gate_passed"] = gate_passed
                out_row["keep_reason"] = row.get("_keep_reason") or (
                    "gate_passed" if gate_passed else "unknown"
                )
            clean.append(out_row)
        buckets[label] = clean
        gate_meta[label] = {
            "top": top, "baseline": baseline, "gate_passed": gate_passed,
            "lexical_evidence": has_lexical, "cut_count": cut_count, "error": sub_error,
        }
        if rerank_info is not None:
            gate_meta[label]["rerank"] = rerank_info  # augment 관측: gate_passed ∧ rerank 잎에서만.
    return buckets, gate_meta


def get_client(url: str | None = None) -> Any:
    """검색용 OpenSearch 클라이언트 — 020 ``opensearch_sync.get_client`` 재사용(단일 출처)."""
    from src.search.opensearch_sync import get_client as _sync_get_client

    return _sync_get_client(url)
