"""관계 품질 순수 메트릭 (spec 031 T003·T004·T005).

LLM/DB 0·결정적. 단계 분리 측정으로 "후보 단계 문제 vs LLM 판단 문제"를 진단한다.
- `candidate_recall`(T003): 후보 단계가 정답 파트너를 회수하는지(대칭 인정).
- `relation_metrics`(T004): 제안 엣지의 precision/recall·kind/고립 정확도.
- `threshold_sweep`(T005): 임계를 쓸어 P/R 곡선(스냅샷 위 결정적 재측정).
"""
from __future__ import annotations

from src.relations.quality.snapshot import ProposedEdge


def isolated_candidates(
    registered_ids: set[str],
    candidate_ids: set[str],
) -> list[str]:
    """후보(`candidate_ids`)에 한 번도 등장하지 않은 registered 자산 = 고립 후보(051 FR-101).

    순수 집합 차 — `candidate_ids` 의 **의미는 호출자가 정한다**. 051 curate 는 부트스트랩 쌍
    (고conf graph_edge + path_signal)에 등장한 자산을 candidate 로 보아 "관계 0 ∧ path 0" 을
    고립으로 정의(035 isolation 의미·관계 단계). 학습 0.

    Args:
        registered_ids: 측정 모집단(등록된 전체 자산 id).
        candidate_ids: 후보로 한 번이라도 등장한 자산 id. **무엇을 후보로 볼지는 호출자가 정한다.**

    Returns:
        고립 자산 id 리스트(정렬 — 결정적).
    """
    return sorted(registered_ids - candidate_ids)


def candidate_recall(
    pairs: list[tuple[str, str]],
    source_candidates: dict[str, set[str]],
) -> float:
    """골든 쌍 중 후보 단계가 파트너를 회수한 비율(FR-003).

    쌍 (a,b)는 `b∈cand[a]` 또는 `a∈cand[b]`면 회수(대칭 kind 양방향 인정).

    Args:
        pairs: 정답 자산 쌍 목록.
        source_candidates: ``{소스 자산: 후보 자산 집합}``. 키가 없으면 후보 없음으로 본다.

    Returns:
        회수 비율 0.0~1.0. **골든이 비면 0.0**(1.0 아님 — 측정 불가를 만점으로 오해하지 않도록).
    """
    if not pairs:
        return 0.0
    hit = 0
    for a, b in pairs:
        if b in source_candidates.get(a, set()) or a in source_candidates.get(b, set()):
            hit += 1
    return hit / len(pairs)


def _accepted_edges(proposed: dict[str, list[ProposedEdge]], confidence_min: float):
    """confidence_min 이상(accepted)인 ``(소스, 엣지)`` 쌍만 골라 흘려보낸다(제너레이터).

    Args:
        proposed: ``{소스 자산: 제안 엣지 목록}``.
        confidence_min: 신뢰도 하한(**이상**이면 통과).

    Yields:
        ``(소스 자산 id, ProposedEdge)``.
    """
    for src, edges in proposed.items():
        for e in edges:
            if e.confidence >= confidence_min:
                yield src, e


def relation_metrics(
    *,
    triples: list[tuple[str, str, str]],
    isolated: set[str],
    proposed: dict[str, list[ProposedEdge]],
    confidence_min: float,
) -> dict:
    """제안 엣지 대비 골든으로 P/R·kind·고립 정확도를 계산한다(FR-004).

    - precision: accepted 엣지 중 골든 쌍과 일치하는 비율(무순 쌍).
    - recall: 골든 쌍 중 accepted 엣지가 덮은 비율(대칭 양방향 인정).
    - kind_accuracy: 일치한 쌍 중 제안 kind가 정답과 같은 비율.
    - isolation_accuracy: 고립 자산 중 accepted 엣지 0인 비율(감사 #2 진단).

    Args:
        triples: 정답 ``(a, b, kind)`` 목록. 쌍은 **무순**으로 비교한다.
        isolated: 관계가 없어야 하는 자산 집합.
        proposed: ``{소스 자산: 제안 엣지 목록}``(측정 대상).
        confidence_min: 이 값 **이상**인 제안만 채택(accepted)으로 본다.

    Returns:
        precision·recall·kind_accuracy·isolation_accuracy + 모수(``n_pairs``·``n_isolated``·
        ``n_accepted``) dict. 분모가 0인 지표는 0.0 이다.
    """
    golden_pairs = {frozenset((a, b)): kind for a, b, kind in triples}
    accepted = list(_accepted_edges(proposed, confidence_min))
    tp_edges = matched_kind = 0
    covered: set[frozenset] = set()
    for src, e in accepted:
        key = frozenset((src, e.target))
        if key in golden_pairs:
            tp_edges += 1
            covered.add(key)
            if e.kind == golden_pairs[key]:
                matched_kind += 1
    precision = (tp_edges / len(accepted)) if accepted else 0.0
    recall = (len(covered) / len(golden_pairs)) if golden_pairs else 0.0
    kind_accuracy = (matched_kind / tp_edges) if tp_edges else 0.0
    iso_clean = sum(
        1 for a in isolated
        if not any(e.confidence >= confidence_min for e in proposed.get(a, [])))
    isolation_accuracy = (iso_clean / len(isolated)) if isolated else 0.0
    return {
        "precision": precision, "recall": recall, "kind_accuracy": kind_accuracy,
        "isolation_accuracy": isolation_accuracy,
        "n_pairs": len(golden_pairs), "n_isolated": len(isolated),
        "n_accepted": len(accepted)}


def min_sim_sweep(
    golden_pairs: list[tuple[str, str]],
    candidates_by_source: dict[str, list[tuple[str, float]]],
    *,
    thresholds: list[float],
) -> list[dict]:
    """min_sim(후보 코사인 유사도) 하한 스윕 — 각 하한에서 recall·통과 후보 수(033 FR-004·N1).

    각 하한 t 에서 `emb_score >= t` 후보만 남기고(소스별 (id, emb_score) 리스트), 남은 후보를
    `dict[str, set[str]]` 로 접어 `candidate_recall`(대칭 인정·DRY) 로 recall 을 잰다.

    Args:
        golden_pairs: 정답 자산 쌍 목록.
        candidates_by_source: ``{소스: [(후보 id, emb_score), ...]}``.
        thresholds: 시험할 하한 값들(0~1). 순서대로 결과 행이 나온다.

    Returns:
        ``[{min_sim, recall, candidates}]`` — 하한을 올릴수록 recall 은 떨어지고 후보 수는 준다.
        이 곡선에서 운영 하한을 고른다.
    """
    rows = []
    for t in thresholds:
        kept: dict[str, set[str]] = {}
        n_cand = 0
        for src, cands in candidates_by_source.items():
            survivors = {tid for tid, score in cands if score >= t}
            kept[src] = survivors
            n_cand += len(survivors)
        rows.append({
            "min_sim": t,
            "recall": candidate_recall(golden_pairs, kept),
            "candidates": n_cand,
        })
    return rows


def auto_approve_sweep(
    golden_pairs: list[tuple[str, str]],
    proposed: dict[str, list[ProposedEdge]],
    *,
    conf_thresholds: list[float],
    emb_thresholds: list[float],
) -> list[dict]:
    """2D(conf×emb) 자동승인 스윕 — 각 격자점의 precision·승인 수(033 FR-005·#3).

    자동승인 집합 = `confidence >= conf_min AND emb_score >= emb_min` 인 제안 엣지.
    precision = 골든 무순 쌍과 일치한 승인 수 / 전체 승인 수(승인 0 이면 0.0).

    Args:
        golden_pairs: 정답 자산 쌍 목록.
        proposed: ``{소스: 제안 엣지 목록}``.
        conf_thresholds: 시험할 LLM 신뢰도 하한 값들.
        emb_thresholds: 시험할 임베딩 유사도 하한 값들.

    Returns:
        격자점마다 ``{conf_min, emb_min, precision, approved}`` 한 행. 행 수는 두 목록 길이의 곱.
    """
    golden_keys = {frozenset((a, b)) for a, b in golden_pairs}
    rows = []
    for c in conf_thresholds:
        for e in emb_thresholds:
            approved = 0
            hit = 0
            for src, edges in proposed.items():
                for edge in edges:
                    if edge.confidence >= c and edge.emb_score >= e:
                        approved += 1
                        if frozenset((src, edge.target)) in golden_keys:
                            hit += 1
            precision = (hit / approved) if approved else 0.0
            rows.append({
                "conf_min": c, "emb_min": e,
                "precision": precision, "approved": approved,
            })
    return rows


def threshold_sweep(
    *,
    triples: list[tuple[str, str, str]],
    isolated: set[str],
    proposed: dict[str, list[ProposedEdge]],
    thresholds: list[float],
) -> list[dict]:
    """각 임계에서 `relation_metrics`를 재사용해 P/R 곡선을 반환한다(FR-006·DRY).

    동결 스냅샷을 입력하면 LLM 을 다시 부르지 않고도 임계만 바꿔 재측정할 수 있다.

    Args:
        triples: 정답 ``(a, b, kind)`` 목록.
        isolated: 관계가 없어야 하는 자산 집합.
        proposed: ``{소스: 제안 엣지 목록}``.
        thresholds: 시험할 신뢰도 하한 값들.

    Returns:
        임계마다 ``{confidence_min, ...relation_metrics 결과}`` 한 행.
    """
    return [
        {"confidence_min": t,
         **relation_metrics(triples=triples, isolated=isolated,
                            proposed=proposed, confidence_min=t)}
        for t in thresholds]
