"""044 — 필드 evidence score 순수 함수(헌법 3조 · LLM 0).

``matched_queries``(OpenSearch named query _name) 기반 strong/weak 가중 합산.
임계·가중치 기본값은 ``src.config.search_constants`` 단일 출처.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from src.config import search_constants as sc
from src.search.query_plan import SearchPolicy

# named query _name → weight(044 plan §Evidence weights).
EVIDENCE_WEIGHTS: dict[str, float] = {
    "hit_keywords": sc.EVIDENCE_HIT_KEYWORDS_WEIGHT,
    "hit_labels": sc.EVIDENCE_HIT_LABELS_WEIGHT,
    "hit_file_name": sc.EVIDENCE_HIT_FILE_NAME_WEIGHT,
    "hit_summary": sc.EVIDENCE_HIT_SUMMARY_WEIGHT,
    "hit_cross_meta": sc.EVIDENCE_HIT_CROSS_META_WEIGHT,
}

_STRONG_EVIDENCE_NAMES: frozenset[str] = frozenset(
    {"hit_keywords", "hit_labels", "hit_file_name"}
)


def evidence_score(
    matched_queries: Collection[str] | None,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """어느 필드에서 단어가 맞았는지를 가중 합산해 "증거 점수"를 낸다(순수·결정적).

    연속적인 BM25 점수가 아니라 **맞았다/아니다 1/0** 에 가중치를 곱해 더한다 — 점수 분포가
    질의마다 달라도 임계가 흔들리지 않게 하려는 설계다.

    Args:
        matched_queries: OpenSearch 가 돌려준 named query 이름들(``hit_keywords`` 등).
            ``None`` 이나 빈 값이면 0.0.
        weights: 이름→가중치 맵. ``None`` 이면 기본 가중치(설정 단일 출처)를 쓴다.
            모르는 이름은 **조용히 무시**한다.

    Returns:
        증거 점수 합. strong 필드(keywords·labels·file_name)가 하나라도 맞았으면
        ``hit_cross_meta`` 가중은 더하지 않는다 — 같은 증거를 두 번 세지 않기 위해서다.
    """
    w = weights or EVIDENCE_WEIGHTS
    names = list(matched_queries or [])
    has_strong = any(n in _STRONG_EVIDENCE_NAMES for n in names if n in w)
    total = 0.0
    for name in names:
        weight = w.get(name)
        if weight is None:
            continue
        if name == "hit_cross_meta" and has_strong:
            continue
        total += weight
    return total


def strong_evidence_score(
    matched_queries: Collection[str] | None,
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """**strong 등급 필드만** 가중 합산한다(keywords·labels·file_name).

    본문·교차 메타(summary·cross_meta)는 우연히 겹치는 일이 잦아 제외한다 — "의도적으로 붙인
    메타에서 맞았는가"만 보는 더 엄격한 신호다.

    Args:
        matched_queries: 맞은 named query 이름들. ``None`` 이면 0.0.
        weights: 이름→가중치 맵. ``None`` 이면 기본값.

    Returns:
        strong 필드 가중치 합.
    """
    w = weights or EVIDENCE_WEIGHTS
    total = 0.0
    for name in matched_queries or []:
        if name in _STRONG_EVIDENCE_NAMES and name in w:
            total += w[name]
    return total


def lexical_rescue_keep(
    matched_queries: Collection[str] | None,
    *,
    policy: SearchPolicy,
    rescue_enabled: bool,
    weights: Mapping[str, float] | None = None,
) -> tuple[bool, str]:
    """의미 유사도 문턱에서 떨어진 행 **한 건**을 단어 증거만으로 살릴지 판정한다(순수·결정적).

    Args:
        matched_queries: 그 행에서 맞은 named query 이름들. ``None`` 은 "관측 자체가 없음"
            (구 색인·mock)이라 **살린다** — 빈 리스트(관측했으나 아무것도 안 맞음)와 구분된다.
        policy: 검색 정책. ``mode='keyword'`` 면 더 관대한 임계를 쓴다.
        rescue_enabled: 끄면 판정 없이 **무조건 살린다**(예전 동작 보존용 토글).
        weights: 증거 가중치 맵. ``None`` 이면 기본값.

    Returns:
        ``(살릴지 여부, 사유 문자열)``. 사유는 로그·디버깅용이며 ``evidence_keyword``·
        ``evidence_normal``·``dropped_weak``·``dropped_no_evidence``·``legacy_*`` 중 하나다.
    """
    if not rescue_enabled:
        return True, "legacy_lexical"

    if matched_queries is None:
        # mock·구 hit( matched_queries 키 없음) — 관측 전 호환 keep.
        return True, "legacy_no_matched_queries"

    mq = list(matched_queries)
    if not mq:
        return False, "dropped_no_evidence"

    ev = evidence_score(mq, weights=weights)

    # 2026-07-24 mode 슬림: restricted(generic+auto) 분기 제거 — keyword=관대 임계 / 그 외=일반 임계.
    if policy.mode == "keyword":
        if ev >= sc.EVIDENCE_KEYWORD_THRESHOLD:
            return True, "evidence_keyword"
        return False, "dropped_weak"

    if ev >= sc.EVIDENCE_NORMAL_THRESHOLD:
        return True, "evidence_normal"
    return False, "dropped_weak"


__all__ = [
    "EVIDENCE_WEIGHTS",
    "evidence_score",
    "lexical_rescue_keep",
    "strong_evidence_score",
]
