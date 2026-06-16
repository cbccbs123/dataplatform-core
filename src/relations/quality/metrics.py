"""관계 품질 순수 메트릭 (spec 031 T003·T004·T005).

LLM/DB 0·결정적. 단계 분리 측정으로 "후보 단계 문제 vs LLM 판단 문제"를 진단한다.
- `candidate_recall`(T003): 후보 단계가 정답 파트너를 회수하는지(대칭 인정).
- `relation_metrics`(T004): 제안 엣지의 precision/recall·kind/고립 정확도.
- `threshold_sweep`(T005): 임계를 쓸어 P/R 곡선(스냅샷 위 결정적 재측정).
"""
from __future__ import annotations

from src.relations.quality.snapshot import ProposedEdge


def candidate_recall(
    pairs: list[tuple[str, str]],
    source_candidates: dict[str, set[str]],
) -> float:
    """골든 쌍 중 후보 단계가 파트너를 회수한 비율(FR-003).

    쌍 (a,b)는 `b∈cand[a]` 또는 `a∈cand[b]`면 회수(대칭 kind 양방향 인정).
    빈 골든은 0.0.
    """
    if not pairs:
        return 0.0
    hit = 0
    for a, b in pairs:
        if b in source_candidates.get(a, set()) or a in source_candidates.get(b, set()):
            hit += 1
    return hit / len(pairs)


def _accepted_edges(proposed: dict[str, list[ProposedEdge]], confidence_min: float):
    """confidence_min 이상(accepted)인 (소스, 엣지) 쌍만 산출한다."""
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
