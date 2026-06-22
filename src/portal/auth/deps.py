"""FastAPI Depends — Bearer 파싱·검증·Principal (spec 042)."""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, Header, HTTPException

from src.portal.auth.config import load_portal_auth_config
from src.portal.auth.principal import ANONYMOUS, Principal, claims_to_principal
from src.portal.auth.verifier import get_token_verifier


def parse_bearer_token(authorization: str | None) -> str | None:
    """``Authorization: Bearer <token>`` 에서 raw token 추출."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Authorization 형식 오류(Bearer 필요)")
    return parts[1].strip() or None


def authenticate_token(token: str) -> Principal:
    """검증기 + ``claims_to_principal``. 실패 시 HTTP 401."""
    try:
        claims = get_token_verifier().verify(token)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰") from exc
    try:
        return claims_to_principal(claims)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def get_principal(
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    """보호 라우트 principal — clearance 가 ``project_ext_meta`` 키 제거 판정에 쓰인다.

    ``PORTAL_AUTH_DISABLED=1``: 토큰 없음 → anonymous(public), Bearer 있으면 검증.
    비활성 아님: Bearer 필수.
    """
    cfg = load_portal_auth_config()
    token = parse_bearer_token(authorization)
    if cfg.auth_disabled:
        if token:
            return authenticate_token(token)
        return ANONYMOUS
    if not token:
        raise HTTPException(status_code=401, detail="인증 필요")
    return authenticate_token(token)


def require_principal(
    principal: Annotated[Principal, Depends(get_principal)],
) -> Principal:
    return principal
