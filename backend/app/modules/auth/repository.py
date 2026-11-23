import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.modules.auth.models import AccountStatus, RefreshSession, User


class AuthRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_user_by_id(self, user_id: uuid.UUID) -> User | None:
        stmt = select(User).where(User.id == user_id)
        return self.db.scalar(stmt)

    def get_user_by_email(self, email: str) -> User | None:
        normalized = email.strip().lower()
        stmt = select(User).where(User.email == normalized)
        return self.db.scalar(stmt)

    def create_user(self, *, email: str, password_hash: str) -> User:
        user = User(
            email=email.strip().lower(),
            password_hash=password_hash,
            status=AccountStatus.ACTIVE.value,
        )
        self.db.add(user)
        self.db.flush()
        return user

    def create_refresh_session(
        self,
        *,
        user_id: uuid.UUID,
        token_hash: str,
        expires_at: datetime,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> RefreshSession:
        session = RefreshSession(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
            user_agent=user_agent[:255] if user_agent else None,
            ip_address=ip_address[:64] if ip_address else None,
        )
        self.db.add(session)
        self.db.flush()
        return session

    def get_refresh_session_for_update(self, token_hash: str) -> RefreshSession | None:
        stmt = (
            select(RefreshSession).where(RefreshSession.token_hash == token_hash).with_for_update()
        )
        return self.db.scalar(stmt)

    def revoke_all_user_sessions(
        self,
        user_id: uuid.UUID,
        *,
        revoked_at: datetime | None = None,
    ) -> None:
        now = revoked_at or datetime.now(UTC)
        stmt = (
            update(RefreshSession)
            .where(
                RefreshSession.user_id == user_id,
                RefreshSession.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        self.db.execute(stmt)
