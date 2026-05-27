"""v2 정책 엔진 — 도메인 팩을 컴포지션 시점에 검증한다.

constraint 는 팩이 고른 전략의 capability 태그를 검사한다. 단계 B는 NoExternalLLM
(외부 LLM 전략 금지)만 의료에 적용한다. PHI 선행·결정성 스코어러·Negative Override 등
cross-asset/PHI 관련 constraint 는 단계 C에서 추가한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.pipeline.packs import DomainPack
    from src.pipeline.registry import StrategyRegistry


class PolicyViolation(Exception):
    """팩이 정책 constraint 를 위반."""


class Constraint(Protocol):
    def check(self, pack: DomainPack, registry: StrategyRegistry) -> str | None:
        """위반 시 사유 문자열, 통과 시 None."""
        ...


@dataclass(frozen=True)
class ForbidTag:
    """팩의 어느 슬롯 전략도 이 태그를 가지면 안 된다."""

    tag: str

    def check(self, pack, registry) -> str | None:
        for slot, name in pack.per_asset.items():
            if self.tag in registry.tags(slot, name):
                return f"{slot}={name} 전략이 금지 태그 '{self.tag}' 보유"
        return None


@dataclass(frozen=True)
class RequireTag:
    """특정 슬롯 전략이 이 태그를 반드시 가져야 한다."""

    slot: str
    tag: str

    def check(self, pack, registry) -> str | None:
        name = pack.per_asset.get(self.slot)
        if name is None:
            return f"슬롯 '{self.slot}' 미정의"
        if self.tag not in registry.tags(self.slot, name):
            return f"{self.slot}={name} 전략에 필수 태그 '{self.tag}' 없음"
        return None


@dataclass(frozen=True)
class DomainPolicy:
    name: str
    constraints: tuple[Constraint, ...] = ()


POLICIES: dict[str, DomainPolicy] = {
    "general_default": DomainPolicy("general_default", ()),
    "medical_strict": DomainPolicy("medical_strict", (ForbidTag("external_llm"),)),
}


def validate(pack: DomainPack, registry: StrategyRegistry) -> None:
    """팩의 정책을 검증한다. 위반 시 PolicyViolation."""
    policy = POLICIES.get(pack.policy)
    if policy is None:
        raise PolicyViolation(f"미등록 정책: {pack.policy!r}")
    violations = [msg for c in policy.constraints if (msg := c.check(pack, registry)) is not None]
    if violations:
        raise PolicyViolation(f"정책 '{policy.name}' 위반: " + "; ".join(violations))
