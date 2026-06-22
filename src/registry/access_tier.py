"""ext_meta 필드별 접근 등급(access_tier) — 순수 상수·합성 (spec 040 Wave 1)."""

from __future__ import annotations

from typing import Literal

AccessTier = Literal["public", "authenticated", "authorized", "regulated"]

PUBLIC: AccessTier = "public"
AUTHENTICATED: AccessTier = "authenticated"
AUTHORIZED: AccessTier = "authorized"
REGULATED: AccessTier = "regulated"

TIER_ORDER: tuple[AccessTier, ...] = (PUBLIC, AUTHENTICATED, AUTHORIZED, REGULATED)

_DOMAIN_FLOORS: dict[str, AccessTier] = {
    "medical": REGULATED,
    "review": AUTHORIZED,
}


def domain_floor(domain: str) -> AccessTier | None:
    """도메인별 최소 access_tier 바닥. general 등 미등록 도메인은 ``None``."""
    return _DOMAIN_FLOORS.get(domain)


def max_tier(*tiers: str | None) -> str:
    """ordinal이 가장 높은 tier 반환. ``None`` 은 무시."""
    present = [t for t in tiers if t]
    if not present:
        raise ValueError("비어 있는 tier 목록")
    return max(present, key=lambda t: TIER_ORDER.index(t))  # type: ignore[arg-type]
