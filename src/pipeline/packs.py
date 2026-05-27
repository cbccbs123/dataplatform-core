"""v2 도메인 팩 — 슬롯별 전략 선택 + 정책.

단계 B: per-asset 슬롯만. 일반·의료는 동일 전략(by_modality)을 쓰되 정책이 다르다.
도메인별 전략 차이(의료 by_signature/medclip)와 cross_asset 슬롯은 단계 C에서 추가.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainPack:
    name: str
    per_asset: dict[str, str]   # 슬롯 → 전략 이름
    policy: str                 # POLICIES 키


GENERAL_PACK = DomainPack(
    name="general",
    per_asset={"classify": "cascade_v1", "extract": "by_modality", "embed": "by_modality", "persist": "asset_upsert"},
    policy="general_default",
)
MEDICAL_PACK = DomainPack(
    name="medical",
    per_asset={"classify": "cascade_v1", "extract": "by_modality", "embed": "by_modality", "persist": "asset_upsert"},
    policy="medical_strict",
)

_PACKS: dict[str, DomainPack] = {"general": GENERAL_PACK, "medical": MEDICAL_PACK}


def for_domain(label: str) -> DomainPack:
    """분류 라벨 → 도메인 팩. 미지정/review 는 일반으로 보수적 폴백."""
    return _PACKS.get(label, GENERAL_PACK)
