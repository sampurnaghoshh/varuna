"""The app boots and /health answers even with its dependencies down (§8).

This test deliberately does not require postgres or redis: a health endpoint
that only works when everything works is not a health endpoint.
"""

from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_200_regardless_of_dependencies() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_health_carries_the_response_envelope() -> None:
    with TestClient(app) as client:
        body = client.get("/api/v1/health").json()
    assert body["source"] == "live"
    assert isinstance(body["elapsed_ms"], int)


def test_health_reports_each_dependency_honestly() -> None:
    with TestClient(app) as client:
        body = client.get("/api/v1/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["services"]) == {"postgis", "redis"}
    healthy = body["services"]["postgis"]["up"] and body["services"]["redis"]["up"]
    assert body["status"] == ("ok" if healthy else "degraded")
