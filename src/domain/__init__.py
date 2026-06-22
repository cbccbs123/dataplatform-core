"""도메인 닫힌 어휘 패키지 (spec 042 ``status_vocab``).

FSM·CHECK 컬럼용 StrEnum 정본. 비즈니스 로직은 각 소비 모듈
(``access_tier`` · ``ingest.status`` · relations 등)에 둔다.
"""

from src.domain.status_vocab import (
    AccessTier,
    GraphEdgeStatus,
    RegistryFieldStatus,
    RelationResolutionStatus,
)

__all__ = [
    "AccessTier",
    "GraphEdgeStatus",
    "RegistryFieldStatus",
    "RelationResolutionStatus",
]
