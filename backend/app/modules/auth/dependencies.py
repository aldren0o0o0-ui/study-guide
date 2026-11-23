from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.exceptions import ForbiddenException, UnauthorizedException
from app.database.session import get_db
from app.modules.auth.models import AccountStatus, User
from app.modules.auth.repository import AuthRepository
from app.modules.auth.security import decode_access_token

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise UnauthorizedException(
            "Authentication credentials were not provided",
            code="AUTHENTICATION_REQUIRED",
        )

    token = credentials.credentials.strip()
    if not token:
        raise UnauthorizedException(
            "Authentication credentials were not provided",
            code="AUTHENTICATION_REQUIRED",
        )

    user_id = decode_access_token(token)
    repo = AuthRepository(db)
    user = repo.get_user_by_id(user_id)

    if user is None:
        raise UnauthorizedException(
            "Authenticated user no longer exists",
            code="USER_NOT_FOUND",
        )

    if user.status != AccountStatus.ACTIVE.value:
        raise ForbiddenException(
            "Your account is not active",
            code="ACCOUNT_INACTIVE",
        )

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
