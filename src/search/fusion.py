"""순위 기반 융합 — Reciprocal Rank Fusion (Cormack et al. 2009).

여러 랭킹 리스트를 점수 크기가 아닌 '순위'로 합친다: RRF(d) = Σ_i 1/(k + rank_i(d)).
스케일·alpha 무관, 결정적(동점은 항목 id 오름차순 tiebreak). 학습 없음(헌법 1·3조).
컷오프 용도로 쓰지 말 것 — RRF 점수는 적합도 절대값이 아니다(설계 §5).
"""
from __future__ import annotations

from collections.abc import Sequence

RRF_DEFAULT_K = 60


def fuse_rrf(
    ranked_lists: Sequence[Sequence[str]], *, k: int = RRF_DEFAULT_K
) -> list[tuple[str, float]]:
    """랭킹 리스트들을 RRF로 융합.

    인자:
        ranked_lists: 각 리스트는 항목 id를 1위→하위 순으로 담은 시퀀스(리스트별 id 유일 가정).
        k: 평활 상수(기본 60). 클수록 최상위 영향력이 완만해진다.
    반환:
        (id, rrf_score) 를 점수 내림차순, 동점은 id 오름차순으로 정렬한 리스트.
    """
    if k <= 0:
        raise ValueError("RRF k는 양수여야 합니다.")
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    # 결정성(헌법 3조): 점수 내림차순, 동점은 id 오름차순.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
