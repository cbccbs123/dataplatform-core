"""포탈 인증 공개 API (spec 042 · 010 US4).

    ``config`` / ``verifier`` / ``principal`` / ``dev_issuer`` / ``deps`` / ``schemas``
"""

from src.portal.auth.deps import authenticate_token, get_principal, require_principal
from src.portal.auth.dev_issuer import issue_dev_token
from src.portal.auth.principal import ANONYMOUS, Principal, claims_to_principal

issue_token = issue_dev_token
decode_token = authenticate_token

__all__ = [
    "ANONYMOUS",
    "Principal",
    "authenticate_token",
    "claims_to_principal",
    "decode_token",
    "get_principal",
    "issue_dev_token",
    "issue_token",
    "require_principal",
]
