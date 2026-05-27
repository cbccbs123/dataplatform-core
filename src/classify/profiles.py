"""도메인 분류 프로파일 — 도메인 지식(시그니처·어휘·LLM 라벨)을 데이터로 분리.

cascade 엔진은 등록 메커니즘이 아니라 ``DomainProfileProvider`` 추상 seam 에만
의존한다. A 단계는 코드 등록(``RegistryProvider``), B 단계(후속)는 DB provider 로
교체하되 엔진·테스트는 불변. 순수 데이터(lexicon/llm_label)는 후일 DB 이전 대상,
signatures 는 능력 모듈(코드)로 유지(설계 원칙 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class SigHit:
    """시그니처 매칭 결과. signature=판별 종류, detail=부가 정보(예: resourceType)."""

    signature: str
    detail: dict[str, Any] = field(default_factory=dict)


# 첫 head 바이트를 받아 매칭 시 SigHit, 아니면 None. 텍스트 디코딩은 규칙 내부 책임.
SignatureRule = Callable[[bytes], "SigHit | None"]


@dataclass(frozen=True)
class DomainProfile:
    """한 도메인의 분류 지식. domain=asset.domain_label 값."""

    domain: str
    lexicon: frozenset[str] = frozenset()      # stage2: 어휘(순수 데이터)
    llm_label: str = ""                         # stage3: LLM 허용 라벨에 합류(순수 데이터)
    signatures: tuple[SignatureRule, ...] = ()  # stage1: 능력 모듈(코드)


class DomainProfileProvider(Protocol):
    def all_profiles(self) -> list[DomainProfile]: ...


# 프로세스 전역 등록(import 부수효과; pipeline.builtins.register_defaults 패턴과 동형).
DOMAIN_PROFILES: dict[str, DomainProfile] = {}


def register_profile(profile: DomainProfile) -> None:
    DOMAIN_PROFILES[profile.domain] = profile


class RegistryProvider:
    """A 단계 provider — 코드로 등록된 프로파일 반환. B 단계에서 DB provider 로 교체."""

    def all_profiles(self) -> list[DomainProfile]:
        return list(DOMAIN_PROFILES.values())
