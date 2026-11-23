import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.config import Settings, parse_cors_origins
from app.core.exceptions import (
    BadRequestException,
    ConflictException,
    ForbiddenException,
    NotFoundException,
    UnauthorizedException,
    register_exception_handlers,
)
from app.core.logging import RequestContextMiddleware
from app.database.base import Base


class SampleInput(BaseModel):
    count: int


def _build_test_error_client() -> TestClient:
    test_app = FastAPI()
    test_app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(test_app)

    @test_app.get("/bad-request")
    def trigger_bad_request() -> None:
        raise BadRequestException("Invalid parameter", code="INVALID_PARAM")

    @test_app.get("/unauthorized")
    def trigger_unauthorized() -> None:
        raise UnauthorizedException()

    @test_app.get("/forbidden")
    def trigger_forbidden() -> None:
        raise ForbiddenException()

    @test_app.get("/not-found")
    def trigger_not_found() -> None:
        raise NotFoundException("Item missing")

    @test_app.get("/conflict")
    def trigger_conflict() -> None:
        raise ConflictException("Duplicate item")

    @test_app.post("/validate")
    def trigger_validation(payload: SampleInput) -> dict[str, int]:
        return {"count": payload.count}

    @test_app.get("/crash")
    def trigger_crash() -> None:
        raise RuntimeError("Unexpected boom")

    return TestClient(test_app, raise_server_exceptions=False)


def test_parse_cors_origins_from_csv_and_list() -> None:
    assert parse_cors_origins("") == []
    assert parse_cors_origins("http://a.com, http://b.com ") == ["http://a.com", "http://b.com"]
    assert parse_cors_origins('["http://a.com", "http://b.com"]') == [
        "http://a.com",
        "http://b.com",
    ]
    assert parse_cors_origins(["http://a.com"]) == ["http://a.com"]


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8081,http://localhost:19006")
    settings = Settings()
    assert settings.API_V1_PREFIX == "/api/v1"
    assert settings.APP_NAME == "StudyGuide API"
    assert settings.CORS_ORIGINS == ["http://localhost:8081", "http://localhost:19006"]


def test_app_exceptions_return_standardized_envelope() -> None:
    client = _build_test_error_client()

    res_400 = client.get("/bad-request")
    assert res_400.status_code == 400
    body_400 = res_400.json()
    assert body_400["success"] is False
    assert body_400["error"]["code"] == "INVALID_PARAM"
    assert body_400["error"]["message"] == "Invalid parameter"
    assert "request_id" in body_400["error"]

    assert client.get("/unauthorized").status_code == 401
    assert client.get("/forbidden").status_code == 403
    assert client.get("/not-found").status_code == 404
    assert client.get("/conflict").status_code == 409


def test_validation_error_returns_field_details() -> None:
    client = _build_test_error_client()
    res = client.post("/validate", json={"count": "not-an-int"})
    assert res.status_code == 422
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert isinstance(body["error"]["details"], list)
    assert body["error"]["details"][0]["field"] == "count"


def test_unhandled_exception_returns_500_without_leaking_internals() -> None:
    client = _build_test_error_client()
    res = client.get("/crash")
    assert res.status_code == 500
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "Unexpected boom" not in body["error"]["message"]


def test_declarative_base_naming_convention() -> None:
    assert Base.metadata.naming_convention["pk"] == "pk_%(table_name)s"
    assert Base.metadata.naming_convention["ix"] == "ix_%(column_0_label)s"
