"""검색 결과의 **융합·게이트·컷 수학** — 전부 순수 함수(OpenSearch 없이 단위 검증 가능).

모달리티마다 [벡터 kNN + BM25] 두 결과를 클라이언트에서 min-max 정규화 + 가중평균으로 합치고
(``fuse_hybrid``), kNN 코사인 표본 하나로 게이트 신호(``gate_signal``)와 per-result 컷(``cut_rows``)을
함께 얻는다. 이 계산을 서버 파이프라인이 아닌 코드로 들고 있는 이유는 재현성 때문이다 — 서버 상태에
따라 결과가 달라지지 않고, 같은 입력이면 언제나 같은 순위가 나온다.

**흐름에서의 위치**: ``opensearch_search`` 가 OpenSearch 를 호출해 hit 을 받아 오면, 이 모듈이 그것을
행으로 바꾸고 합쳐서 걸러낸다. 검색 본문 조립은 ``query_builder`` 가 맡는다.

``opensearch_search`` 는 이 모듈의 함수들을 **재export** 한다 — 그래서 테스트·호출부가
``opensearch_search.passes_cutoff`` 처럼 참조하거나 patch 하는 코드가 있다(같은 함수다).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    """어떤 값이든 **유한한 실수**로 바꾼다(결정적·순수).

    NaN·무한대는 비교가 비결정적이라(정렬 순서가 흔들린다) 여기서 전부 걸러낸다.

    Args:
        value: 숫자·문자열·``None`` 무엇이든.
        default: 변환 실패나 비유한 값일 때 쓸 대체값.

    Returns:
        유한 실수.
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def knn_score_to_cosine(score: Any) -> float:
    """lucene cosinesimil knn ``_score``(=(1+cos)/2)를 원시 코사인으로 환산한다(순수·결정적).

    020 인덱스 매핑이 ``space_type=cosinesimil``·``engine=lucene`` 이므로 lucene 의 코사인 knn 점수는
    ``(1+cos)/2`` 다 → ``cos = 2·score − 1``. 비유한·범위 밖은 [-1,1] 로 안전 clamp. 실 OS 의 정확한
    스케일·calibration 은 G4 실OS 에서 확정한다(여기선 환산식만 고정).

    비유한 _score 의 안전 기본값은 0.0(코사인 -1)이 아니라 **0.5(코사인 0, 중립)**다 — 환산식이
    ``score=0.5`` 에서 ``cos=0`` 이라, 무효 점수는 게이트를 끄는 쪽이 아니라 중립으로 떨어뜨려야 한다.

    Args:
        score: OpenSearch knn ``_score``. 숫자가 아니어도 받는다.

    Returns:
        [-1, 1] 로 clamp 된 코사인 유사도.
    """
    cos = 2.0 * _safe_float(score, 0.5) - 1.0
    return max(-1.0, min(1.0, cos))


# 게이트 경계 포용(>=) 부동소수 허용오차. 임계가 0.10·0.65 처럼 십진 근사라 0.75-0.65=0.0999…998
# 같은 표현오차로 **경계값이 탈락**하지 않게 한다(결정적·관측 무영향한 미소값).
_CUTOFF_TOL = 1e-9


def passes_cutoff(top: float, baseline: float, *, eps: float, floor: float) -> bool:
    """그 모달리티 버킷을 유지할지 판정한다(순수·결정적).

    ``유지 = (top − baseline) ≥ eps AND top ≥ floor``. 상대 신호가 주판정이고 절대 하한은 코퍼스
    전체가 평평할 때를 위한 backstop 이다. 부동소수 표현오차로 경계값이 탈락하지 않도록 미세
    허용오차를 두고 비교한다.

    Args:
        top: 그 버킷 kNN 표본의 최고 코사인.
        baseline: 배경 수준(``gate_signal`` 이 주는 하위 절반 평균).
        eps: ``top − baseline`` 이 이만큼은 벌어져야 한다는 **상대** 기준. 주판정이다.
        floor: ``top`` 자체의 **절대** 하한. 코퍼스가 통째로 평평할 때를 위한 backstop.

    Returns:
        버킷을 유지하면 True. **둘 다 만족해야 한다**(AND) — 하나라도 못 넘으면 그 버킷은 빈
        결과로 접혀 무관한 항목이 노출되지 않는다.
    """
    t = _safe_float(top)
    return (t - _safe_float(baseline)) >= eps - _CUTOFF_TOL and t >= floor - _CUTOFF_TOL


def minmax_normalize(scores: Iterable[float]) -> list[float]:
    """점수 리스트를 [0,1] 로 min-max 정규화한다(순수·결정적, 027 FR-002).

    클라이언트 융합의 1단계다 — OS 서버 normalization-processor 가 하던 min-max 를 **순수 함수로
    이관**(헌법 3조: 융합 수학을 서버 상태가 아닌 단위 검증 가능한 코드로). 빈 입력은 ``[]``,
    Args:
        scores: 정규화할 점수들. 숫자가 아닌 값도 안전하게 0.0 으로 흡수한다.

    Returns:
        같은 순서·같은 길이의 [0,1] 값 리스트(호출부가 hit 리스트와 zip 한다). 빈 입력은 ``[]``,
        **전부 같은 값이면 전원 1.0** — 0 으로 나누지 않으면서 기존 순위를 그대로 둔다.
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
    """kNN 코사인 표본에서 게이트 신호 ``(최고값, 배경 수준)`` 을 뽑는다(순수·결정적).

    배경 수준을 **하위 절반 평균**으로 잡는 이유: 전체 평균을 쓰면 관련 자산이 많은 질의('충전'처럼
    비슷한 영상이 여럿)에서 배경이 끌려 올라가 최고값과의 격차가 좁아지고, 멀쩡한 버킷이 게이트에
    걸린다. 하위 절반만 보면 "무관한 꼬리"의 수준이 유지된다.

    Args:
        cosines: 그 버킷 kNN 표본의 원시 코사인들(순서 무관).

    Returns:
        ``(top, baseline)``. 표본이 2개 미만이면 하위 절반을 만들 수 없어 ``baseline=0.0``,
        빈 표본이면 ``(0.0, 0.0)``.
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

    Args:
        rows: 이미 융합·정렬된 행들. ``_bm25``(어휘 증거 여부)·``_cos``(원시 코사인) 내부 키를 본다.
        result_floor: 코사인 하한. 이 값 미만이고 어휘 증거도 없으면 버린다.

    Returns:
        살아남은 행(입력 순서 보존 — **랭킹은 바꾸지 않고** 꼬리만 잘라낸다).
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
    """게이트·컷을 통과한 행의 **순서만** cross-encoder 로 다시 매긴다(순수·결정적).

    걸러내는 일은 하지 않는다 — 선별은 ``cut_rows`` 몫이고 여기는 재정렬 전담이다(리랭크가 선별까지
    맡았을 때 정답 행이 잘려 회수율이 떨어진 실측이 있다).

    **융합 점수 값 자체는 보존하고 순서만 바꾼다.** 리랭크 점수(0~1 절대값)로 덮어쓰면 다른 버킷의
    융합 점수(상대값)와 척도가 어긋나 모달리티 간 순위가 망가지기 때문이다. 그래서 상위 행들이 갖고
    있던 융합 점수 집합을 그대로 두고, 리랭크가 매긴 순서대로 **재배정**한다.

    Args:
        rows: 게이트·컷을 통과한 행들(융합 순서).
        query: 질의 문자열(리랭커 입력).
        rerank_fn: 채점 함수 **주입 seam**. ``None`` 이면 기본 cross-encoder 를 지연 import 한다
            — 리랭크가 꺼진 환경에 무거운 의존을 끌어오지 않기 위해서다.
        top_r: 앞에서 몇 행만 채점할지(지연 통제). 나머지는 융합 순서 그대로 뒤에 붙는다.
        tau: 이 점수 미만인 head 행을 **드롭**한다. **기본 0.0 = 드롭 없음**(순수 재정렬) —
            실측에서 τ 드롭이 recall 을 깎았기 때문이다.
        model_name: 리랭커 모델 이름.

    Returns:
        ``(재정렬된 행, head 채점값)``. head 가 비면 입력을 그대로 돌려주고 채점하지 않는다.
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
    """질의 정규화를 켜고 끄는 순수 토글 함수(기본 꺼짐).

    자기 자신은 정규화 로직을 갖지 않는다 — 켜져 있을 때 ``llm_fn`` 을 부를 뿐이다. 콜백의 결정성은
    호출부가 보장한다.

    Args:
        query: 원본 질의.
        enabled: 꺼져 있으면(기본) **원문을 바이트 그대로** 돌려준다.
        llm_fn: 정규화 콜백 주입 seam. 이름은 LLM 이지만 **형태소 정규화 같은 비-LLM 콜백**도
            받는다(파라미터명은 하위호환으로 유지). ``None`` 이면 정규화하지 않는다.

    Returns:
        정규화된 질의, 또는 원문(꺼짐·빈 질의·콜백 미주입).
    """
    if not enabled or not query or llm_fn is None:
        return query
    return llm_fn(query)


def os_hit_to_row(hit: dict[str, Any]) -> dict[str, Any]:
    """OpenSearch hit 을 버킷 행 dict 로 옮긴다(순수·결정적).

    자산 식별 키가 ``asset_id`` 가 아니라 **``id``** 라는 점만 주의하면 된다 — 버킷 행 계약이 그렇게
    정해져 있어 색인 문서의 ``asset_id`` 를 이 이름으로 옮긴다. 여기서 넣는 ``similarity`` 는
    서브검색 단독 점수이고, 이후 ``fuse_hybrid`` 가 융합 점수로 덮어쓴다.

    Args:
        hit: OpenSearch 응답의 hit 하나. ``_source`` 가 없거나 필드가 비어도 안전하게 처리한다.

    Returns:
        버킷 행 dict. ``matched_queries`` 는 응답에 있을 때만 넣는다 — **키 자체의 유무**가
        "관측했는가"를 뜻하기 때문에 빈 리스트로 채우면 안 된다(rescue 판정이 달라진다).
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
        # 057-후속: 결과-좁히기 패싯·클라 필터용 주제(색인된 keyword). 필터 terms{topics} 와 **동일 소스**
        # 라 facet 약속과 클릭 결과가 일치한다(라이브 투영 대비 불일치·N+1 제거). OS 문서에 이미 있어 무비용.
        "topics": [str(t) for t in (src.get("topics") or [])],
        "subtopics": [str(t) for t in (src.get("subtopics") or [])],
        # 059 FR-104: 색인된 topic_pairs("topic>subtopic" 짝)를 행에 전달한다 — 프론트가 topic→subtopic
        # 트리를 교차곱 오배치 없이 그리게(하위호환 필드·미존재 시 [] 폴백). 평면 topics/subtopics 와
        # 동일 소스(_topics_doc_fields)라 표시·패싯 전용·랭킹 미반영(무회귀).
        "topic_pairs": [str(t) for t in (src.get("topic_pairs") or [])],
    }
    mq = hit.get("matched_queries")
    if mq is not None:
        row["matched_queries"] = [str(x) for x in mq]
    return row


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
    Args:
        bm25_hits: BM25 서브검색 hit 들.
        knn_hits: kNN 서브검색 hit 들.
        weights: ``(w_bm25, w_knn)`` 가중치. 두 축의 정규화 점수를 이 비율로 섞는다.

    Returns:
        융합·정렬된 행 리스트(점수 내림차순, 동점은 id 오름차순으로 **결정적**). 각 행에는 게이트·
        컷이 쓰는 내부 키 ``_cos``(kNN 코사인·없으면 None)·``_bm25``(어휘 매칭 여부)가 붙는다.
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
        """hit 을 행으로 바꿔 ``merged`` 에 등록하고(이미 있으면 재사용) 그 항목을 돌려준다.

        같은 자산이 BM25·kNN 양쪽에 나오면 **한 행으로 합쳐야** 하므로, 여기서 id 기준으로
        모아 두고 점수만 각 루프가 채운다. 리랭크·증거 필터가 쓸 내부 키(``_rrtext``·``_about``·
        ``_kwtext``)도 이 시점에 붙인다(응답 직전에 제거된다).
        """
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
        # 073: aboutness OR-증거 필터 내부키 — _about(적재시 확정 개체)·_kwtext(keywords+파일명 합본).
        # bucket_policy 의 about_or_filter 가 증거 매칭에 쓰고, clean 이 응답 전 제거한다(_rrtext 동형).
        row["_about"] = [str(a) for a in (src_.get("about") or [])]
        row["_kwtext"] = " ".join(
            [*(str(x) for x in _kw), str(src_.get("fs_uri") or "").split("/")[-1]]
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
