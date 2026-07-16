from collections.abc import Iterator
from typing import cast

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from pytest import MonkeyPatch

from app.core.config import get_settings
from main import create_app


@pytest.fixture
def cors_client(monkeypatch: MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://web.example.com")
    get_settings.cache_clear()
    with TestClient(create_app()) as client:
        yield client
    get_settings.cache_clear()


def preflight(client: TestClient, origin: str) -> Response:
    return cast(
        Response,
        client.options(
            "/market/status",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        ),
    )


def test_preflight_allows_configured_origin_without_credentials(
    cors_client: TestClient,
) -> None:
    response = preflight(cors_client, "https://web.example.com")

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://web.example.com"
    assert response.headers["access-control-allow-methods"] == "GET, OPTIONS"
    assert "access-control-allow-credentials" not in response.headers


def test_preflight_rejects_disallowed_origin(cors_client: TestClient) -> None:
    response = preflight(cors_client, "https://other.example.com")

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
