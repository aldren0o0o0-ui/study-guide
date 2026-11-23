import uuid
from datetime import timedelta

import jwt
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import UnauthorizedException
from app.modules.auth.security import (
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)


def test_argon2id_password_hashing_and_verification() -> None:
    hashed = hash_password("StrongPass123")
    assert hashed.startswith("$argon2id$")
    assert hashed != "StrongPass123"
    assert verify_password("StrongPass123", hashed) is True
    assert verify_password("WrongPass123", hashed) is False


def test_refresh_token_generation_and_hashing() -> None:
    raw_1 = generate_refresh_token()
    raw_2 = generate_refresh_token()
    assert raw_1 != raw_2
    assert len(raw_1) >= 48

    digest = hash_refresh_token(raw_1)
    assert len(digest) == 64
    assert digest != raw_1
    assert hash_refresh_token(raw_1) == digest


def test_access_token_creation_and_decoding() -> None:
    user_id = uuid.uuid4()
    token, expires_in = create_access_token(user_id)
    assert expires_in == 15 * 60
    decoded_id = decode_access_token(token)
    assert decoded_id == user_id


def test_expired_access_token_raises_unauthorized() -> None:
    user_id = uuid.uuid4()
    token, _ = create_access_token(user_id, expires_delta=timedelta(seconds=-10))
    with pytest.raises(UnauthorizedException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.code == "ACCESS_TOKEN_EXPIRED"


def test_access_token_with_wrong_issuer_or_type_rejected() -> None:
    settings = Settings()
    bad_type_token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "iss": settings.JWT_ISSUER,
            "aud": settings.JWT_AUDIENCE,
            "iat": 1700000000,
            "exp": 4700000000,
            "jti": str(uuid.uuid4()),
            "type": "refresh",
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(UnauthorizedException) as exc_info:
        decode_access_token(bad_type_token, settings=settings)
    assert exc_info.value.code == "INVALID_ACCESS_TOKEN"


def test_production_settings_reject_insecure_jwt_secret() -> None:
    with pytest.raises(ValidationError):
        Settings(APP_ENV="production", JWT_SECRET_KEY="short-secret")

    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="production",
            JWT_SECRET_KEY="dev-only-insecure-jwt-secret-key-change-in-production-32chars",
        )

    valid_prod = Settings(
        APP_ENV="production",
        JWT_SECRET_KEY="a-very-strong-cryptographic-production-secret-key-2026",
    )
    assert valid_prod.APP_ENV == "production"
