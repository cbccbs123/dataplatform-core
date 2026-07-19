"""도메인 닫힌 어휘 패키지 (spec 042 ``status_vocab``).

FSM·CHECK 컬럼용 StrEnum 정본은 ``src.domain.status_vocab`` 에 있다
(``AccessTier`` · ``GraphEdgeStatus`` · ``RegistryFieldStatus`` · ``RelationResolutionStatus``).
소비 모듈은 전부 ``from src.domain.status_vocab import ...`` 로 직접 import 하므로 여기서는
재수출하지 않는다(069 US-F 에서 소비 0 재수출 제거). 비즈니스 로직은 각 소비 모듈
(``access_tier`` · ``ingest.status`` · relations 등)에 둔다.
"""
