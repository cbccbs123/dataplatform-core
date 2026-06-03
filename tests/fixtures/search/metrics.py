"""검색 평가 지표(이진 적합도, 순수 함수). ranked=id 순위 리스트, relevant=정답 id 집합."""
from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in relevant)
    return hit / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in relevant)
    return hit / k


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    # 표준 이진 nDCG@k. IDCG는 '전체 정답 수'(min(len(relevant), k)) 기준이라
    # top-k 안에 못 들어온 정답도 분모에 반영된다 → 저조한 recall 을 패널티(관례적 정의).
    # 결정적 산술만 사용(헌법 3조).
    dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(ranked[:k], start=1) if x in relevant)
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0
