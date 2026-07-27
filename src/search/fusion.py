"""OpenSearch 클라이언트 **융합·게이트·컷 수학** (검색 read path, spec 027·029).

069 US-E(FR-E5): 종전 ``opensearch_search`` 한 파일에서 **순수 융합/게이트/컷 수학**만 분리했다.
여기의 함수는 OS·opensearch-py 없이 **순수·결정적**으로 동작하며 단위 게이트에서 항상 검증된다
(헌법 3조). 본문 빌더는 ``query_builder``, IO(실행)는 ``opensearch_search`` 가 담당한다.

027 클라이언트 융합: OS 서버 normalization-processor(min-max + arithmetic_mean)를 순수 함수로 이관.
모달리티당 [plain kNN + BM25] 를 클라이언트가 min-max+가중평균으로 융합(``fuse_hybrid``)하고, kNN 원시
코사인에서 게이트 신호(``gate_signal``)·per-result 컷(``cut_rows``)을 같은 표본 1회로 얻는다.

하위호환: 기존 ``opensearch_search.<name>`` import·patch 경로는 그 모듈이 이 심볼들을 **재export**해
그대로 유지된다(US-E patch seam 보존).
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
    """적합도 게이트(순수·결정적, 023 FR-001·004 계승): 그 modality 버킷을 유지할지 판정한다.

    ``keep = (top − baseline) ≥ eps AND top ≥ floor``. 상대 신호 ``top − baseline``(background 흡수)이
    주판정, 절대 ``floor`` 는 코퍼스 전체가 평평할 때의 느슨한 backstop. 둘 다 만족해야 유지(AND);
    실패면 빈 버킷(no-match → 무관 결과 표출 차단). 비교는 ``_CUTOFF_TOL`` 로 경계를 포용한다.

    027: ``baseline`` 은 ``gate_signal`` 이 주는 **하위 절반 평균**(robust)이다 — 023 의 전체 평균을
    대체해 밀집 토픽('충전')에서 baseline 이 끌려 올라가는 오컷을 구조 해결한다.

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
    """kNN 원시 코사인 리스트에서 게이트 신호 ``(top, robust baseline)`` 을 뽑는다(순수·결정적, FR-003).

    023 probe(추가 kNN 호출)를 대체한다 — 융합이 클라이언트로 오면서 kNN 응답에 원시 코사인이
    자연히 보존되므로, **추가 검색 없이** 같은 kNN 표본에서 신호를 잰다(SC-002: probe 소멸).

    - ``top`` = 표본 최댓값(그 modality 의 최적합 신호).
    - ``baseline`` = **코사인 하위 절반 평균**(robust). 023 의 '전체 평균' 은 밀집 토픽(예: '충전'
      → 다수 영상이 고코사인)에서 baseline 이 끌어올려져 ``top−baseline`` 이 좁아지는 오컷을 낳았다.
      하위 절반(정렬 후 floor(n/2)개)만 평균내면 background(무관 꼬리) 해석이 유지돼 밀집 토픽에서도
      신호가 분리된다 — 신규 분기·상수 0 의 **구조 해결**(027 핵심).

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
    """검색 질의를 LLM 핵심 명사구로 정규화하는 순수 토글 함수(029 FR-004, 기본 off).

    028 후속 측정이 확인한 명사구 정규화('별 보는 방법'→'천체 관측' 0.009→0.227, 패러프레이즈
    오컷 직격)를 **설정 토글**로 들인다. 021 FR-004(검색시점 LLM 금지)를 거버넌스 절차로 개정한
    뒤에만 켜며(G2), 본 함수는 그 토글의 **순수 잎**이다 — ``llm_fn`` 주입 seam·네트워크 0·결정적.

    072: ``llm_fn`` 은 LLM 전용이 아니라 **임의의 정규화 콜백**(nori 형태소 명사추출 등)을 받는다 —
    파라미터명은 하위호환으로 유지하나 실제 주입체는 형태소 정규화(검색시점 LLM 0·결정적)일 수 있다.

    - ``enabled=False``(기본): 원문 그대로 반환한다 — **바이트 동일 passthrough**(027 회귀 0의 봉인
      지점). 빈/``None`` 질의도 정규화할 내용이 없으므로 그대로 반환한다(llm_fn 미호출 안전).
    - ``enabled=True``: ``llm_fn(query)`` 가 돌려준 명사구를 반환한다. ``llm_fn`` 의 결정성(temp=0·
      env 입력 0)은 호출부(G2 ``query_norm`` seam)가 보장한다 — 본 함수는 같은 입력에 같은 출력.
      방어적으로 ``llm_fn`` 미주입(``None``)이면 정규화 수단이 없으므로 원문을 그대로 둔다(배선
      누락이 결정성을 깨지 않게 — fail-safe to 027).

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
