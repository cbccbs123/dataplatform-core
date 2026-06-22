"""dev 전용 토큰 발급 (spec 042 MVP).

``POST /auth/token`` 전용 — 비밀번호 검증 없음, 로컬 스모크용.
운영 IdP 연동 시 본 모듈·엔드포인트는 비활성화 예정.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt

from src.portal.auth.config import load_portal_auth_config


def issue_dev_token(*, user_id: str) -> str:
    """로컬 HS256 JWT — ``LocalHs256Verifier`` 와 secret·알고리즘 쌍을 맞춘다."""
    cfg = load_portal_auth_config()
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(seconds=cfg.jwt_ttl_seconds),
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm="HS256")
