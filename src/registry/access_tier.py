"""ext_meta 필드별 접근 등급(access_tier) — 순수 상수·합성·read projection (spec 040·042).

040-W1 — DB ``ext_meta_field_registry.access_tier`` 정의·시드.
042    — ``project_ext_meta`` 로 포탈 read path **키 제거(omit)** — clearance × field tier × domain_floor.

정책(plan D2): 값을 null·마스킹 문자열로 바꾸지 **않고**, clearance 미달 키를 응답 dict 에서 **빼다**.
write path(ingest)는 본 모듈을 사용하지 않음 — DB 에는 전량 적재.
"""

from __future__ import annotations

from src.domain.status_vocab import AccessTier

PUBLIC = AccessTier.PUBLIC
AUTHENTICATED = AccessTier.AUTHENTICATED
AUTHORIZED = AccessTier.AUTHORIZED
REGULATED = AccessTier.REGULATED

# ordinal 낮→높. tier_allows·max_tier·effective_field_tier 가 이 순서에 의존.
TIER_ORDER: tuple[AccessTier, ...] = (
    AccessTier.PUBLIC,
    AccessTier.AUTHENTICATED,
    AccessTier.AUTHORIZED,
    AccessTier.REGULATED,
)

# 도메인 바닥 — 필드 tier 와 합성 시 더 높은 쪽이 effective_field_tier(042 read omit 판정).
_DOMAIN_FLOORS: dict[str, AccessTier] = {
    "medical": REGULATED,
    "review": AUTHORIZED,
}


def domain_floor(domain: str) -> AccessTier | None:
    """도메인별 최소 access_tier 바닥. general 등 미등록 도메인은 ``None``."""
    return _DOMAIN_FLOORS.get(domain)


def max_tier(*tiers: str | AccessTier | None) -> str:
    """ordinal이 가장 높은 tier 문자열 반환. ``None`` 은 무시."""
    present = [AccessTier(t) if not isinstance(t, AccessTier) else t for t in tiers if t]
    if not present:
        raise ValueError("비어 있는 tier 목록")
    return max(present, key=lambda t: TIER_ORDER.index(t)).value


def principal_clearance(*, authenticated: bool) -> str:
    """요청자의 열람 등급을 정한다(현재 2단계).

    Args:
        authenticated: 인증을 통과했는지.

    Returns:
        등급 문자열. ⚠️ **가장 높은 등급은 여기서 부여하지 않는다** — 그 등급이 필요한 도메인은
        도메인 자체의 하한으로 읽기를 막는다. 역할 기반 권한은 계정 체계가 들어온 뒤에 다룬다.
    """
    return AUTHORIZED.value if authenticated else PUBLIC.value


def effective_field_tier(field_tier: str, domain: str) -> str:
    """레지스트리 필드 tier 와 ``domain_floor`` 합성 — ordinal 높은 쪽."""
    floor = domain_floor(domain)
    if floor is None:
        return field_tier
    return max_tier(field_tier, floor.value)


def tier_allows(clearance: str, required: str) -> bool:
    """``clearance`` ordinal ≥ ``required`` 이면 read 노출 허용."""
    return TIER_ORDER.index(AccessTier(clearance)) >= TIER_ORDER.index(AccessTier(required))


def project_ext_meta(
    ext_meta: dict | None,
    field_tiers: dict[str, str],
    *,
    domain: str,
    clearance: str,
) -> dict:
    """read path ext_meta projection (042) — **키 제거(omit)**, null 치환 아님.

    clearance 미달 키는 출력 dict 에 넣지 않는다(plan D2).
    레지스트리 미등록 키는 통과(레거시·커스텀 키 보존).

    Args:
        ext_meta: DB ``asset_metadata.ext_meta`` 원본(또는 검색 DTO 부분집합).
        field_tiers: ``fetch_access_tiers(conn, domain)``.
        domain: ``asset.domain_label`` — ``domain_floor`` 합성.
        clearance: ``Principal.clearance``.
    """
    if not ext_meta:
        return {}
    out: dict = {}
    for key, value in ext_meta.items():
        tier = field_tiers.get(key)
        if tier is None:
            # 041 레지스트리 미등록 키 — 레거시·커스텀 보존(ingest 039 키 게이트와 별개).
            out[key] = value
            continue
        if tier_allows(clearance, effective_field_tier(tier, domain)):
            out[key] = value
    return out
