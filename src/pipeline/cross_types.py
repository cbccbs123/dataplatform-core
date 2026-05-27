"""cross-asset 파이프라인 스테이지 간 타입(후보→점수→결정→증거).

현 relations 경로는 dict(list[dict])를 쓴다. 단계 A에서 타입만 정의하고,
실제 배선은 단계 C(공용 그래프 + 의료 ER)에서 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    source_id: str
    target_id: str
    block_key: str = ""        # 어떤 블로킹 키로 잡혔나(추적). embedding_topk 면 'embedding'.
    method: str = ""           # 'blocking' | 'embedding_topk'


@dataclass(frozen=True)
class Evidence:
    field: str                 # 비교 필드명
    comparator: str            # exact | jaro_winkler | embedding_cosine | llm_zs ...
    similarity: float | None = None
    m_prob: float | None = None
    u_prob: float | None = None
    weight: float = 0.0        # log(m/u)


@dataclass(frozen=True)
class ScoredPair:
    candidate: Candidate
    score: float
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    candidate: Candidate
    verdict: str               # match | review | non_match
    score: float
    overrides: list[str] = field(default_factory=list)  # 예: ['negative_override:dob']
