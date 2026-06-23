"""포탈 인증 설정 — env 단일 출처 (spec 042).

환경변수
    PORTAL_AUTH_DISABLED   — 1 이면 dev bypass(anonymous 허용)
    PORTAL_AUTH_BACKEND    — 현재 ``local_hs256`` 만
    PORTAL_JWT_SECRET      — HS256 서명 키(dev bypass 시 기본값 허용·운영 전 교체)
    PORTAL_JWT_TTL_SECONDS — dev ``/auth/token`` 발급 수명(기본 3600)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_DEV_SECRET = "dev-portal-jwt-change-in-prod"
_VALID_BACKENDS = frozenset({"local_hs256"})


@dataclass(frozen=True)
class PortalAuthConfig:
    """포탈 API 인증 설정 스냅샷."""

    auth_disabled: bool
    backend: str
    jwt_secret: str
    jwt_ttl_seconds: int


def _env_bool(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def load_portal_auth_config() -> PortalAuthConfig:
    """환경변수에서 ``PortalAuthConfig`` 를 읽는다."""
    backend = os.getenv("PORTAL_AUTH_BACKEND", "local_hs256").strip().lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"지원하지 않는 PORTAL_AUTH_BACKEND={backend!r} "
            f"(지원: {sorted(_VALID_BACKENDS)})"
        )
    auth_disabled = _env_bool("PORTAL_AUTH_DISABLED")
    raw_secret = os.getenv("PORTAL_JWT_SECRET", "").strip()
    if auth_disabled:
        secret = raw_secret or _DEFAULT_DEV_SECRET
    elif not raw_secret:
        raise ValueError(
            "PORTAL_AUTH_DISABLED=0 인데 PORTAL_JWT_SECRET 미설정 — "
            "운영·인증 활성 환경에서는 서명 키가 필요합니다"
        )
    else:
        secret = raw_secret
    raw_ttl = os.getenv("PORTAL_JWT_TTL_SECONDS", "3600").strip()
    try:
        ttl = max(60, int(raw_ttl))
    except ValueError:
        ttl = 3600
    return PortalAuthConfig(
        auth_disabled=auth_disabled,
        backend=backend,
        jwt_secret=secret,
        jwt_ttl_seconds=ttl,
    )
