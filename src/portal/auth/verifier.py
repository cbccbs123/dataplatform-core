"""액세스 토큰 검증 (spec 042).

``TokenVerifier`` Protocol — backend 교체 시 ``verify`` 구현만 추가.
현재: ``LocalHs256Verifier`` (dev ``/auth/token`` 과 동일 secret·HS256).

발급은 ``dev_issuer`` — 본 모듈은 **검증만**.
"""

from __future__ import annotations

from typing import Any, Protocol

import jwt

from src.portal.auth.config import PortalAuthConfig, load_portal_auth_config


class TokenVerifier(Protocol):
    """Bearer access token 검증. 성공 시 검증된 claims dict."""

    def verify(self, token: str) -> dict[str, Any]: ...


class LocalHs256Verifier:
    """dev/MVP — 포탈 자체 HS256(secret 공유)."""

    def __init__(self, config: PortalAuthConfig) -> None:
        self._secret = config.jwt_secret

    def verify(self, token: str) -> dict[str, Any]:
        return jwt.decode(token, self._secret, algorithms=["HS256"])


_verifier: TokenVerifier | None = None  # 프로세스 내 싱글턴 — secret·backend 변경은 재기동 전제.


def get_token_verifier(*, config: PortalAuthConfig | None = None) -> TokenVerifier:
    """``PortalAuthConfig.backend`` 에 맞는 검증기 싱글턴."""
    global _verifier
    if _verifier is not None and config is None:
        return _verifier
    cfg = config or load_portal_auth_config()
    if cfg.backend == "local_hs256":
        _verifier = LocalHs256Verifier(cfg)
        return _verifier
    raise ValueError(f"미구현 auth backend: {cfg.backend!r}")


def _reset_verifier_for_tests() -> None:
    """단위 테스트용 — 검증기 캐시 초기화."""
    global _verifier
    _verifier = None
