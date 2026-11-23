from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.modules.auth.dependencies import CurrentUser
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
from app.modules.auth.service import AuthService
from app.shared.responses import ApiResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _extract_client_metadata(request: Request) -> tuple[str | None, str | None]:
    user_agent = request.headers.get("User-Agent")
    ip_address = request.client.host if request.client else None
    return user_agent, ip_address


@router.post(
    "/register",
    response_model=ApiResponse[AuthResponseData],
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: RegisterRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[AuthResponseData]:
    user_agent, ip_address = _extract_client_metadata(request)
    service = AuthService(db)
    data = service.register_user(payload, user_agent=user_agent, ip_address=ip_address)
    return ApiResponse(data=data, message="Registration successful")


@router.post(
    "/login",
    response_model=ApiResponse[AuthResponseData],
    status_code=status.HTTP_200_OK,
)
def login(
    payload: LoginRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[AuthResponseData]:
    user_agent, ip_address = _extract_client_metadata(request)
    service = AuthService(db)
    data = service.authenticate_user(payload, user_agent=user_agent, ip_address=ip_address)
    return ApiResponse(data=data, message="Login successful")


@router.post(
    "/refresh",
    response_model=ApiResponse[TokenPair],
    status_code=status.HTTP_200_OK,
)
def refresh_tokens(
    payload: RefreshTokenRequest,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[TokenPair]:
    user_agent, ip_address = _extract_client_metadata(request)
    service = AuthService(db)
    tokens = service.refresh_tokens(payload, user_agent=user_agent, ip_address=ip_address)
    return ApiResponse(data=tokens, message="Tokens refreshed")


@router.post(
    "/logout",
    response_model=ApiResponse[LogoutResponseData],
    status_code=status.HTTP_200_OK,
)
def logout(
    payload: LogoutRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ApiResponse[LogoutResponseData]:
    service = AuthService(db)
    data = service.logout(payload)
    return ApiResponse(data=data, message="Logged out")


@router.get(
    "/me",
    response_model=ApiResponse[UserRead],
    status_code=status.HTTP_200_OK,
)
def get_me(current_user: CurrentUser) -> ApiResponse[UserRead]:
    return ApiResponse(data=UserRead.model_validate(current_user))
