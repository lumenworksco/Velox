"""V10 Auth — JWT token creation and validation for web dashboard.

Configuration via environment variables:
    DASHBOARD_SECRET_KEY: JWT signing key (required for auth)
    DASHBOARD_USERNAME: Login username (default: admin)
    DASHBOARD_PASSWORD_HASH: bcrypt hash of password

Auth is ENABLED by default. To explicitly disable (dev only):
    DASHBOARD_AUTH_DISABLED=true
"""

import os
import logging
import hashlib
import hmac
import json
import base64
import time
from functools import wraps

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY", "")
USERNAME = os.getenv("DASHBOARD_USERNAME", "admin")
PASSWORD_HASH = os.getenv("DASHBOARD_PASSWORD_HASH", "")
TOKEN_EXPIRY_HOURS = int(os.getenv("DASHBOARD_TOKEN_EXPIRY_HOURS", "24"))

# Auth is ON by default — must explicitly disable with DASHBOARD_AUTH_DISABLED=true
_auth_disabled = os.getenv("DASHBOARD_AUTH_DISABLED", "false").lower() == "true"
AUTH_ENABLED = not _auth_disabled

if AUTH_ENABLED and not SECRET_KEY:
    logger.warning(
        "DASHBOARD_SECRET_KEY not set — dashboard auth is enabled but no key configured. "
        "Set DASHBOARD_SECRET_KEY or DASHBOARD_AUTH_DISABLED=true"
    )


def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def create_token(username: str) -> str:
    """Create a JWT-like token (HS256)."""
    header = _b64_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64_encode(json.dumps({
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_EXPIRY_HOURS * 3600,
    }).encode())

    signing_input = f"{header}.{payload}"
    signature = hmac.new(SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
    sig_b64 = _b64_encode(signature)

    return f"{header}.{payload}.{sig_b64}"


def verify_token(token: str) -> dict | None:
    """Verify a JWT-like token. Returns payload dict or None if invalid."""
    if not AUTH_ENABLED:
        return {"sub": "anonymous"}

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, sig_b64 = parts

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(SECRET_KEY.encode(), signing_input.encode(), hashlib.sha256).digest()
        actual_sig = _b64_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        # Decode payload
        payload = json.loads(_b64_decode(payload_b64))

        # Check expiry
        if payload.get("exp", 0) < time.time():
            return None

        return payload
    except Exception:
        return None


def verify_password(password: str) -> bool:
    """Verify a password against the stored bcrypt hash.

    Generate hash: python3 -c "import bcrypt; print(bcrypt.hashpw(b'PASSWORD', bcrypt.gensalt()).decode())"
    """
    if not PASSWORD_HASH:
        return False

    import bcrypt
    if not (PASSWORD_HASH.startswith("$2b$") or PASSWORD_HASH.startswith("$2a$")):
        logger.error(
            "DASHBOARD_PASSWORD_HASH is not a bcrypt hash. "
            "Generate one: python3 -c \"import bcrypt; print(bcrypt.hashpw(b'PASSWORD', bcrypt.gensalt()).decode())\""
        )
        return False
    return bcrypt.checkpw(password.encode(), PASSWORD_HASH.encode())


def get_fastapi_dependency():
    """Create a FastAPI dependency for route protection.

    Usage in web_dashboard.py:
        from auth.jwt_auth import get_fastapi_dependency
        require_auth = get_fastapi_dependency()

        @app.get("/api/trades")
        async def get_trades(user=Depends(require_auth)):
            ...
    """
    from fastapi import Depends, HTTPException, Header

    async def require_auth(authorization: str = Header(None)):
        if not AUTH_ENABLED:
            return {"sub": "anonymous"}

        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        token = authorization[7:]  # Strip "Bearer "
        payload = verify_token(token)
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        return payload

    return require_auth
