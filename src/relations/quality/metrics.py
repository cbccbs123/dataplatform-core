"""관계 품질 순수 메트릭 (spec 031 T003·T004·T005).

LLM/DB 0·결정적. 단계 분리 측정으로 "후보 단계 문제 vs LLM 판단 문제"를 진단한다.
- `candidate_recall`(T003): 후보 단계가 정답 파트너를 회수하는지(대칭 인정).
- `relation_metrics`(T004): 제안 엣지의 precision/recall·kind/고립 정확도.
- `threshold_sweep`(T005): 임계를 쓸어 P/R 곡선(스냅샷 위 결정적 재측정).
"""
from __future__ import annotations


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
