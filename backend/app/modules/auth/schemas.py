import re
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


def normalize_email_address(raw_email: str) -> str:
    return raw_email.strip().lower()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email_before(cls, value: object) -> object:
        if isinstance(value, str):
            return normalize_email_address(value)
        return value

    @field_validator("email", mode="after")
    @classmethod
    def normalize_email_after(cls, value: EmailStr) -> str:
        return normalize_email_address(str(value))

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, value: str) -> str:
        if not re.search(r"[A-Za-z]", value):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"\d", value):
            raise ValueError("Password must contain at least one number")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email_before(cls, value: object) -> object:
        if isinstance(value, str):
            return normalize_email_address(value)
        return value

    @field_validator("email", mode="after")
    @classmethod
    def normalize_email_after(cls, value: EmailStr) -> str:
        return normalize_email_address(str(value))


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=16, max_length=512)


class LogoutRequest(BaseModel):
    refresh_token: str = Field(..., min_length=16, max_length=512)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    status: str
    created_at: datetime
    updated_at: datetime


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int


class AuthResponseData(BaseModel):
    user: UserRead
    tokens: TokenPair


class LogoutResponseData(BaseModel):
    revoked: bool
