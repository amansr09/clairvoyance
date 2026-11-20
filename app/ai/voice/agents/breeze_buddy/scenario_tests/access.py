"""The token an eval session presents for its lead, signed like the try-on
token (assist/commerce/ucp/try_on_policy.py): ``"<expires_at>.<signature>"``,
an HMAC of the lead id and expiry under the server's signing key, so only the
agent-tests worker can open a session for a lead, and only for a short while."""

import hmac
import time
from typing import Optional

from app.core.config.static import JWT_SECRET_KEY
from app.core.security.sha import calculate_hmac_sha256

# A test call connects within seconds of its token being made.
_TOKEN_TTL_SECONDS = 10 * 60


def session_token(lead_id: str, expires_at: Optional[int] = None) -> str:
    exp = (
        expires_at if expires_at is not None else int(time.time()) + _TOKEN_TTL_SECONDS
    )
    signature = calculate_hmac_sha256(f"eval-session:{exp}:{lead_id}", JWT_SECRET_KEY)
    return f"{exp}.{signature}" if signature else ""


def token_matches(lead_id: str, token: Optional[str]) -> bool:
    """The token is client input, so every check here must refuse rather than
    raise: compare_digest raises on non-ASCII str, and int() raises past 4300
    digits. A real expiry is a 10-digit Unix time."""
    if not token or not token.isascii():
        return False
    exp, _, _ = token.partition(".")
    if not exp.isdigit() or len(exp) > 12 or int(exp) < time.time():
        return False
    expected = session_token(lead_id, int(exp))
    return bool(expected) and hmac.compare_digest(expected, token)
