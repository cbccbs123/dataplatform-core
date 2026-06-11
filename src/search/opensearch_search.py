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
    OS_RESULT_FLOOR_DEFAULT,
)
from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind

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
    """BM25 multi_match 단독 서브검색 본문(순수·결정적, 027 FR-001).

    027 클라이언트 융합 전환: 종전 서버 hybrid 쿼리의 텍스트 서브쿼리를 **독립
    검색 본문**으로 떼어낸다 — msearch 의 한 서브검색으로 보내 BM25 원시 ``_score`` 를 그대로 받아
    클라이언트(fuse_hybrid)가 min-max 정규화·가중평균한다(파이프라인 소거). 구성은 026 계승:
    boost 차등 ``_TEXT_FIELDS``(summary^3…file_name^0.5), modality terms 필터, 의료 ``must_not``.

    operator='and' 면 전 nori 토큰 매칭 강제(025 FR-001 — 복합어 가짜매칭 차단). 기본 'or' 는
    operator 키를 생략해 현행 본문과 바이트 동일(회귀 0). ``size`` 는 ``k``(요청 버킷 한도).
    """
    filters: list[dict[str, Any]] = [{"terms": {"modality": sorted(modality_values)}}]
    mm: dict[str, Any] = {"query": query, "fields": list(_TEXT_FIELDS)}
    if operator != "or":
        mm["operator"] = operator
    clause: dict[str, Any] = {"must": [{"multi_match": mm}], "filter": filters}
    if exclude_medical:
        clause["must_not"] = [{"term": {"domain_label": _MEDICAL_LABEL}}]
    return {"size": int(k), "query": {"bool": clause}}


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

    media_search 버킷 행의 핵심 키(``id``·``file_uri``·``modality``·``summary``·``similarity``)에
    맞춘다 — 검색 서비스가 OS 버킷과 PG 버킷을 같은 모양으로 병합할 수 있게 한다(응답 동형).
    ``similarity`` 는 hit 의 원시 ``_score`` 다(서브검색 단독 점수) — 027 클라이언트 융합에서는
    ``fuse_hybrid`` 가 이 값을 **융합 점수로 덮어쓴다**(min-max+가중평균). ``_source`` 누락·메타
    None 도 안전 처리한다.

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
# IO 함수 (027) — opensearch-py·임베더는 모듈 상단이 아니라 **함수 내부에서 지연 import**.
# 플래그 off(pg 백엔드) 환경의 모듈 순수성(상단 import 없음)을 보존하기 위함이다 — 위 순수 함수만
# 쓰는 단위 게이트는 opensearch-py·임베더 미설치여도 import 가능해야 한다(020 동형).
# 실제 OS 동작 검증은 G4(실OS e2e). 여기 IO 는 가짜 msearch 클라이언트로 액션 조립을 단위 검증한다.
# 027 T006: 서버 융합 파이프라인 등록 코어와 별도 게이트 검색(추가 kNN) 경로는 제거됐다 — 융합이
# 클라이언트로 이동해 서버 상태(파이프라인) 0·중복 kNN 0(SC-002).
# ──────────────────────────────────────────────────────────────────────────


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
    검색용 LLM 질의 구조화는 호출하지 않는다(원문 질의를 nori·임베딩에 직접).
    """
    labels = list(modalities)
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
        if not cutoff_enabled:
            # 디버그 우회: 게이트·per-result 컷 모두 off → 융합 전체 노출.
            kept = fused
            gate_passed = True
        else:
            gate_passed = passes_cutoff(top, baseline, eps=cutoff_eps, floor=cutoff_floor)
            if gate_passed:
                kept = cut_rows(fused, result_floor=result_floor)
            elif has_lexical and bm25_operator == "and":
                # 어휘 구제(027 정밀화): 코사인 게이트가 실패해도 BM25 증거 행은 살린다 — 약한-있음
                # 토픽(자산 수 적어 코사인 신호 약함·어휘는 정확)이 인접-없음보다 버킷 통계상
                # 약해지는 역전을 행 단위 증거로 해소. 단 ① 의미-노이즈 유입을 막기 위해 구제 시엔
                # **어휘 증거 행만** 남기고(cos-only 배제), ② **operator='and'(전 토큰 매칭)일 때만**
                # 적용한다 — 'or' 매칭은 단일 토큰 우연 일치도 _bm25=True 라 증거 강도가 없어
                # 무관 결과가 구제를 타고 누수된다(리뷰 후속 — 전제의 코드 강제).
                kept = [r for r in fused if r.get("_bm25")]
            else:
                kept = []  # 어휘 증거도 없음 → 빈 버킷(no-match)

        cut_count = len(fused) - len(kept)  # 게이트·컷으로 제거된 행 수(상한 절삭 제외 — 컷 효과만).
        # 내부키(_cos·_bm25) 제거 + 요청 k 상한(응답 동형·구 size=k 계약 보존).
        clean = [
            {key: val for key, val in row.items() if key not in ("_cos", "_bm25")}
            for row in kept[: int(k)]
        ]
        buckets[label] = clean
        gate_meta[label] = {
            "top": top, "baseline": baseline, "gate_passed": gate_passed,
            "lexical_evidence": has_lexical, "cut_count": cut_count, "error": sub_error,
        }
    return buckets, gate_meta


def get_client(url: str | None = None) -> Any:
    """검색용 OpenSearch 클라이언트 — 020 ``opensearch_sync.get_client`` 재사용(단일 출처)."""
    from src.search.opensearch_sync import get_client as _sync_get_client

    return _sync_get_client(url)
