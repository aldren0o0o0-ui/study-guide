import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.core.config import Settings, get_settings
from app.core.exceptions import UnauthorizedException

password_hasher = PasswordHash((Argon2Hasher(),))
_DUMMY_PASSWORD_HASH = password_hasher.hash("dummy-constant-time-fallback-password")


def hash_password(plain_password: str) -> str:
    return password_hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hasher.verify(plain_password, hashed_password)


def verify_dummy_password(plain_password: str) -> None:
    """Runs Argon2id verification against a dummy hash to equalize timing when a user is missing."""
    password_hasher.verify(plain_password, _DUMMY_PASSWORD_HASH)


def generate_refresh_token() -> str:
    """Generates a cryptographically secure 384-bit URL-safe opaque token."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(raw_token: str) -> str:
    """Hashes a raw refresh token with SHA-256 before persistence or lookup."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_access_token(
    user_id: uuid.UUID,
    *,
    settings: Settings | None = None,
    expires_delta: timedelta | None = None,
) -> tuple[str, int]:
    """Creates a signed JWT access token and returns `(token, expires_in_seconds)`."""
    cfg = settings or get_settings()
    now = datetime.now(UTC)
    ttl = (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=cfg.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    expires_at = now + ttl

    payload: dict[str, object] = {
        "sub": str(user_id),
        "iss": cfg.JWT_ISSUER,
        "aud": cfg.JWT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": str(uuid.uuid4()),
        "type": "access",
    }

    encoded = jwt.encode(
        payload,
        cfg.JWT_SECRET_KEY,
        algorithm=cfg.JWT_ALGORITHM,
    )
    return encoded, int(ttl.total_seconds())


def decode_access_token(
    token: str,
    *,
    settings: Settings | None = None,
) -> uuid.UUID:
    """Validates a JWT access token and returns the authenticated user's UUID."""
    cfg = settings or get_settings()
    try:
        payload = jwt.decode(
            token,
            cfg.JWT_SECRET_KEY,
            algorithms=[cfg.JWT_ALGORITHM],
            issuer=cfg.JWT_ISSUER,
            audience=cfg.JWT_AUDIENCE,
            options={"require": ["sub", "iss", "aud", "exp", "iat", "jti", "type"]},
        )
    except ExpiredSignatureError as exc:
        raise UnauthorizedException(
            "Access token has expired",
            code="ACCESS_TOKEN_EXPIRED",
        ) from exc
    except InvalidTokenError as exc:
        raise UnauthorizedException(
            "Invalid access token",
            code="INVALID_ACCESS_TOKEN",
        ) from exc

    if payload.get("type") != "access":
        raise UnauthorizedException(
            "Invalid token type",
            code="INVALID_ACCESS_TOKEN",
        )

    sub = payload.get("sub")
    if not isinstance(sub, str):
        raise UnauthorizedException(
            "Invalid token subject",
            code="INVALID_ACCESS_TOKEN",
        )

    try:
        return uuid.UUID(sub)
    except ValueError as exc:
        raise UnauthorizedException(
            "Malformed token subject",
            code="INVALID_ACCESS_TOKEN",
        ) from exc
