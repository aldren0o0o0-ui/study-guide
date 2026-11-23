import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConflictException,
    ForbiddenException,
    UnauthorizedException,
)
from app.modules.auth.models import AccountStatus, User
from app.modules.auth.repository import AuthRepository
from app.modules.auth.schemas import (
    AuthResponseData,
    LoginRequest,
    LogoutRequest,
    LogoutResponseData,
    RefreshTokenRequest,
    RegisterRequest,
    TokenPair,
    UserRead,
)
from app.modules.auth.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_dummy_password,
    verify_password,
)

logger = logging.getLogger("app.modules.auth")


class AuthService:
    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self.db = db
        self.repo = AuthRepository(db)
        self.settings = settings or get_settings()

    def register_user(
        self,
        payload: RegisterRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> AuthResponseData:
        normalized_email = str(payload.email).strip().lower()
        existing = self.repo.get_user_by_email(normalized_email)
        if existing is not None:
            raise ConflictException(
                "An account with this email address already exists",
                code="EMAIL_ALREADY_REGISTERED",
            )

        hashed_pw = hash_password(payload.password)
        try:
            user = self.repo.create_user(email=normalized_email, password_hash=hashed_pw)
            tokens = self._issue_tokens_for_user(
                user,
                user_agent=user_agent,
                ip_address=ip_address,
            )
            self.db.commit()
            self.db.refresh(user)
        except IntegrityError as exc:
            self.db.rollback()
            raise ConflictException(
                "An account with this email address already exists",
                code="EMAIL_ALREADY_REGISTERED",
            ) from exc

        logger.info(
            "User registered",
            extra={"extra_data": {"user_id": str(user.id)}},
        )
        return AuthResponseData(
            user=UserRead.model_validate(user),
            tokens=tokens,
        )

    def authenticate_user(
        self,
        payload: LoginRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> AuthResponseData:
        normalized_email = str(payload.email).strip().lower()
        user = self.repo.get_user_by_email(normalized_email)

        if user is None:
            verify_dummy_password(payload.password)
            raise UnauthorizedException(
                "Account not found. Please check your email or sign up for a new account.",
                code="USER_NOT_FOUND",
            )

        if not verify_password(payload.password, user.password_hash):
            raise UnauthorizedException(
                "Incorrect password. Please try again.",
                code="INCORRECT_PASSWORD",
            )

        if user.status != AccountStatus.ACTIVE.value:
            raise ForbiddenException(
                "Your account is not active",
                code="ACCOUNT_INACTIVE",
            )

        tokens = self._issue_tokens_for_user(
            user,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        self.db.commit()

        logger.info(
            "User logged in",
            extra={"extra_data": {"user_id": str(user.id)}},
        )
        return AuthResponseData(
            user=UserRead.model_validate(user),
            tokens=tokens,
        )

    def refresh_tokens(
        self,
        payload: RefreshTokenRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> TokenPair:
        raw_token = payload.refresh_token.strip()
        token_hash = hash_refresh_token(raw_token)
        now = datetime.now(UTC)

        session = self.repo.get_refresh_session_for_update(token_hash)
        if session is None:
            raise UnauthorizedException(
                "Invalid refresh token",
                code="INVALID_REFRESH_TOKEN",
            )

        if session.revoked_at is not None:
            if session.replaced_by_session_id is not None:
                # Reuse of an already-rotated token detected: revoke all active sessions for user
                self.repo.revoke_all_user_sessions(session.user_id, revoked_at=now)
                self.db.commit()
                logger.warning(
                    "Refresh token reuse detected; revoked all active user sessions",
                    extra={"extra_data": {"user_id": str(session.user_id)}},
                )
                raise UnauthorizedException(
                    "Refresh token has already been rotated",
                    code="REFRESH_TOKEN_REUSED",
                )
            raise UnauthorizedException(
                "Refresh token has been revoked",
                code="REFRESH_TOKEN_REVOKED",
            )

        expires_at = (
            session.expires_at
            if session.expires_at.tzinfo is not None
            else session.expires_at.replace(tzinfo=UTC)
        )
        if expires_at <= now:
            session.revoked_at = now
            self.db.commit()
            raise UnauthorizedException(
                "Refresh token has expired",
                code="REFRESH_TOKEN_EXPIRED",
            )

        user = self.repo.get_user_by_id(session.user_id)
        if user is None:
            session.revoked_at = now
            self.db.commit()
            raise UnauthorizedException(
                "User account not found",
                code="INVALID_REFRESH_TOKEN",
            )

        if user.status != AccountStatus.ACTIVE.value:
            session.revoked_at = now
            self.db.commit()
            raise ForbiddenException(
                "Your account is not active",
                code="ACCOUNT_INACTIVE",
            )

        # Atomic rotation: create replacement session and mark current session revoked/replaced
        new_raw_refresh = generate_refresh_token()
        new_refresh_hash = hash_refresh_token(new_raw_refresh)
        new_expires_at = now + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS)

        new_session = self.repo.create_refresh_session(
            user_id=user.id,
            token_hash=new_refresh_hash,
            expires_at=new_expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        session.revoked_at = now
        session.replaced_by_session_id = new_session.id

        access_token, expires_in = create_access_token(user.id, settings=self.settings)
        self.db.commit()

        return TokenPair(
            access_token=access_token,
            refresh_token=new_raw_refresh,
            expires_in=expires_in,
        )

    def logout(self, payload: LogoutRequest) -> LogoutResponseData:
        raw_token = payload.refresh_token.strip()
        token_hash = hash_refresh_token(raw_token)
        now = datetime.now(UTC)

        session = self.repo.get_refresh_session_for_update(token_hash)
        if session is not None and session.revoked_at is None:
            session.revoked_at = now
            self.db.commit()
            logger.info(
                "Refresh session revoked on logout",
                extra={"extra_data": {"user_id": str(session.user_id)}},
            )
            return LogoutResponseData(revoked=True)

        return LogoutResponseData(revoked=False)

    def _issue_tokens_for_user(
        self,
        user: User,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> TokenPair:
        now = datetime.now(UTC)
        raw_refresh = generate_refresh_token()
        refresh_hash = hash_refresh_token(raw_refresh)
        expires_at = now + timedelta(days=self.settings.REFRESH_TOKEN_EXPIRE_DAYS)

        self.repo.create_refresh_session(
            user_id=user.id,
            token_hash=refresh_hash,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        access_token, expires_in = create_access_token(user.id, settings=self.settings)
        return TokenPair(
            access_token=access_token,
            refresh_token=raw_refresh,
            expires_in=expires_in,
        )
