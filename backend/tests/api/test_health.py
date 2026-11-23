from unittest.mock import patch

from fastapi.testclient import TestClient


def test_health_endpoint_returns_ok_without_db_check(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "ok"
    assert payload["data"]["service"] == "StudyGuide API"
    assert payload["data"]["database"] == "not_checked"
    assert payload["request_id"] == response.headers["X-Request-ID"]


def test_health_endpoint_preserves_custom_request_id(client: TestClient) -> None:
    custom_req_id = "req-custom-12345"
    response = client.get("/api/v1/health", headers={"X-Request-ID": custom_req_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_req_id
    assert response.json()["request_id"] == custom_req_id


def test_health_endpoint_succeeds_when_database_is_unavailable(client: TestClient) -> None:
    with patch("app.api.v1.health.check_database_status", return_value="unavailable"):
        response = client.get("/api/v1/health?check_db=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "ok"
    assert payload["data"]["database"] == "unavailable"


def test_health_endpoint_reports_database_ok_when_reachable(client: TestClient) -> None:
    with patch("app.api.v1.health.check_database_status", return_value="ok"):
        response = client.get("/api/v1/health?check_db=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["status"] == "ok"
    assert payload["data"]["database"] == "ok"


def test_cors_headers_on_preflight_request(client: TestClient) -> None:
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:8081",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:8081"
