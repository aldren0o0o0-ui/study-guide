from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.auth.models import AccountStatus, RefreshSession, User
from app.modules.auth.security import create_access_token, hash_refresh_token


def test_register_success_and_email_normalization(client: TestClient, db_session: Session) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "  Student.User@Example.COM  ", "password": "SecurePassword123"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True

    user_data = body["data"]["user"]
    tokens = body["data"]["tokens"]

    assert user_data["email"] == "student.user@example.com"
    assert user_data["status"] == "active"
    assert "password" not in user_data
    assert "password_hash" not in user_data
    assert tokens["token_type"] == "bearer"
    assert len(tokens["access_token"]) > 20
    assert len(tokens["refresh_token"]) > 20

    # Verify raw refresh token is NOT stored in PostgreSQL
    stored_session = db_session.scalar(select(RefreshSession))
    assert stored_session is not None
    assert stored_session.token_hash != tokens["refresh_token"]
    assert stored_session.token_hash == hash_refresh_token(tokens["refresh_token"])


def test_register_duplicate_email_rejected(client: TestClient) -> None:
    first = client.post(
        "/api/v1/auth/register",
        json={"email": "dup@example.com", "password": "Password123"},
    )
    assert first.status_code == 201

    second = client.post(
        "/api/v1/auth/register",
        json={"email": "DUP@EXAMPLE.COM", "password": "AnotherPassword123"},
    )
    assert second.status_code == 409
    err = second.json()["error"]
    assert err["code"] == "EMAIL_ALREADY_REGISTERED"


def test_register_invalid_data_rejected(client: TestClient) -> None:
    bad_email = client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": "Password123"},
    )
    assert bad_email.status_code == 422

    weak_pw = client.post(
        "/api/v1/auth/register",
        json={"email": "valid@example.com", "password": "onlyletters"},
    )
    assert weak_pw.status_code == 422

    short_pw = client.post(
        "/api/v1/auth/register",
        json={"email": "valid@example.com", "password": " a1 "},
    )
    assert short_pw.status_code == 422


def test_login_success_and_failure_cases(client: TestClient) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"email": "learner@example.com", "password": "CorrectPassword99"},
    )

    # Successful login with unnormalized email
    login_ok = client.post(
        "/api/v1/auth/login",
        json={"email": "  LEARNER@example.com ", "password": "CorrectPassword99"},
    )
    assert login_ok.status_code == 200
    assert login_ok.json()["data"]["user"]["email"] == "learner@example.com"

    # Wrong password
    wrong_pw = client.post(
        "/api/v1/auth/login",
        json={"email": "learner@example.com", "password": "WrongPassword99"},
    )
    assert wrong_pw.status_code == 401
    assert wrong_pw.json()["error"]["code"] == "INCORRECT_PASSWORD"

    # Unknown account
    unknown = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "CorrectPassword99"},
    )
    assert unknown.status_code == 401
    assert unknown.json()["error"]["code"] == "USER_NOT_FOUND"


def test_protected_me_endpoint_requires_valid_access_token(
    client: TestClient, db_session: Session
) -> None:
    # Unauthenticated request rejected
    unauth = client.get("/api/v1/auth/me")
    assert unauth.status_code == 401
    assert unauth.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"

    # Register and call /me with valid access token
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": "me@example.com", "password": "Password123"},
    )
    access_token = reg.json()["data"]["tokens"]["access_token"]
    user_id = reg.json()["data"]["user"]["id"]

    me_res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me_res.status_code == 200
    assert me_res.json()["data"]["id"] == user_id
    assert me_res.json()["data"]["email"] == "me@example.com"

    # Expired access token rejected
    user_obj = db_session.scalar(select(User).where(User.email == "me@example.com"))
    assert user_obj is not None
    expired_token, _ = create_access_token(
        user_obj.id,
        expires_delta=timedelta(seconds=-30),
    )
    expired_res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert expired_res.status_code == 401
    assert expired_res.json()["error"]["code"] == "ACCESS_TOKEN_EXPIRED"

    # Disabled user rejected
    user_obj.status = AccountStatus.DISABLED.value
    db_session.commit()
    disabled_res = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert disabled_res.status_code == 403
    assert disabled_res.json()["error"]["code"] == "ACCOUNT_INACTIVE"


def test_refresh_token_rotation_and_reuse_detection(
    client: TestClient, db_session: Session
) -> None:
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": "rotate@example.com", "password": "Password123"},
    )
    initial_refresh = reg.json()["data"]["tokens"]["refresh_token"]

    # Rotate token once
    rotate_1 = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": initial_refresh},
    )
    assert rotate_1.status_code == 200
    second_refresh = rotate_1.json()["data"]["refresh_token"]
    assert second_refresh != initial_refresh

    # Reusing the already-rotated `initial_refresh` must fail AND revoke `second_refresh`
    reuse_res = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": initial_refresh},
    )
    assert reuse_res.status_code == 401
    assert reuse_res.json()["error"]["code"] == "REFRESH_TOKEN_REUSED"

    # Now `second_refresh` must also be revoked due to compromise containment
    compromised_res = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": second_refresh},
    )
    assert compromised_res.status_code == 401
    assert compromised_res.json()["error"]["code"] == "REFRESH_TOKEN_REVOKED"

    active_sessions = db_session.scalars(
        select(RefreshSession).where(RefreshSession.revoked_at.is_(None))
    ).all()
    assert len(active_sessions) == 0


def test_logout_revokes_refresh_session_and_rejects_expired_tokens(
    client: TestClient, db_session: Session
) -> None:
    reg = client.post(
        "/api/v1/auth/register",
        json={"email": "logout@example.com", "password": "Password123"},
    )
    refresh_token = reg.json()["data"]["tokens"]["refresh_token"]

    logout_res = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": refresh_token},
    )
    assert logout_res.status_code == 200
    assert logout_res.json()["data"]["revoked"] is True

    # Refreshing with logged-out token fails
    after_logout = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert after_logout.status_code == 401
    assert after_logout.json()["error"]["code"] == "REFRESH_TOKEN_REVOKED"

    # Unknown refresh token fails
    unknown_refresh = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "nonexistent-random-refresh-token-value-12345"},
    )
    assert unknown_refresh.status_code == 401
    assert unknown_refresh.json()["error"]["code"] == "INVALID_REFRESH_TOKEN"

    # Expired refresh token fails
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "logout@example.com", "password": "Password123"},
    )
    expiring_token = login_res.json()["data"]["tokens"]["refresh_token"]
    session_row = db_session.scalar(
        select(RefreshSession).where(
            RefreshSession.token_hash == hash_refresh_token(expiring_token)
        )
    )
    assert session_row is not None
    session_row.expires_at = datetime.now(UTC) - timedelta(minutes=5)
    db_session.commit()

    expired_refresh = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": expiring_token},
    )
    assert expired_refresh.status_code == 401
    assert expired_refresh.json()["error"]["code"] == "REFRESH_TOKEN_EXPIRED"
