import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from app.auth.config import get_auth_settings
from app.auth.dependencies import require_admin
from app.auth.middleware import PUBLIC_ROUTES
from app.auth.tokens import ALGORITHM, AUDIENCE, ISSUER, csrf_digest
from app.core.config import get_settings
from main import create_app

PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$YfDFPLLCSGMA2JVLzUaQCA$"
    "g0pQ2ia+qwLo1MkzAzaquAQB3FVqwxae6Wce+2DDjb8"
)
SECRET = "s" * 32


def configure_auth(monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True) -> None:
    monkeypatch.setenv("ARGOS_AUTH_ENFORCEMENT_ENABLED", str(enabled).lower())
    monkeypatch.setenv("ARGOS_ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ARGOS_ADMIN_PASSWORD_HASH", PASSWORD_HASH)
    monkeypatch.setenv("ARGOS_JWT_SECRET", SECRET)
    monkeypatch.setenv("ARGOS_AUTH_COOKIE_SECURE", "false")
    get_settings.cache_clear()
    get_auth_settings.cache_clear()


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    configure_auth(monkeypatch)
    with TestClient(create_app()) as client:
        yield client
    get_settings.cache_clear()
    get_auth_settings.cache_clear()


def login(client: TestClient) -> None:
    response = client.post(
        "/auth/login", json={"username": "admin", "password": "correct-password"}
    )
    assert response.status_code == 204


def test_login_sets_strict_session_and_csrf_cookies(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/auth/login", json={"username": "admin", "password": "correct-password"}
    )
    cookies = response.headers.get_list("set-cookie")
    session = next(item for item in cookies if item.startswith("argos_session="))
    csrf = next(item for item in cookies if item.startswith("argos_csrf="))
    assert response.status_code == 204
    assert "HttpOnly" in session
    assert "HttpOnly" not in csrf
    assert "SameSite=strict" in session and "Path=/" in session
    assert "SameSite=strict" in csrf and "Path=/" in csrf
    assert response.content == b""


def test_wrong_username_and_password_are_indistinguishable(
    auth_client: TestClient,
) -> None:
    wrong_user = auth_client.post(
        "/auth/login", json={"username": "unknown", "password": "correct-password"}
    )
    wrong_password = auth_client.post(
        "/auth/login", json={"username": "admin", "password": "wrong"}
    )
    assert (
        (wrong_user.status_code, wrong_user.json())
        == (
            wrong_password.status_code,
            wrong_password.json(),
        )
        == (401, {"detail": "Invalid credentials"})
    )


def test_login_schema_is_exact(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/auth/login",
        json={"username": "admin", "password": "correct-password", "extra": True},
    )
    assert response.status_code == 422


def test_me_returns_only_public_session_fields(auth_client: TestClient) -> None:
    login(auth_client)
    response = auth_client.get("/auth/me")
    assert response.status_code == 200
    assert set(response.json()) == {"username", "role", "expires_at"}
    assert response.json()["username"] == "admin"
    assert response.json()["role"] == "admin"


def test_logout_requires_csrf_and_clears_both_cookies(auth_client: TestClient) -> None:
    login(auth_client)
    assert auth_client.post("/auth/logout").status_code == 403
    csrf = auth_client.cookies["argos_csrf"]
    assert (
        auth_client.post(
            "/auth/logout", headers={"X-CSRF-Token": "different"}
        ).status_code
        == 403
    )
    response = auth_client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
    assert response.status_code == 204
    assert "argos_session" not in auth_client.cookies
    assert "argos_csrf" not in auth_client.cookies


def test_logout_rejects_missing_csrf_cookie_with_header(
    auth_client: TestClient,
) -> None:
    login(auth_client)
    csrf = auth_client.cookies["argos_csrf"]
    del auth_client.cookies["argos_csrf"]
    assert (
        auth_client.post("/auth/logout", headers={"X-CSRF-Token": csrf}).status_code
        == 403
    )


@pytest.mark.parametrize(
    "path",
    [
        "/health/database",
        "/market/instruments",
        "/paper/runs",
        "/dashboard/overview",
        "/optimization/refinement-runs",
    ],
)
def test_representative_private_routes_reject_before_domain_work(
    auth_client: TestClient, path: str
) -> None:
    assert auth_client.get(path).status_code == 401


def test_authenticated_request_reaches_business_router(auth_client: TestClient) -> None:
    login(auth_client)
    assert auth_client.get("/market/instruments").status_code == 200


def test_only_exact_health_and_login_are_public(auth_client: TestClient) -> None:
    assert frozenset({("GET", "/health"), ("POST", "/auth/login")}) == PUBLIC_ROUTES
    assert auth_client.get("/health").json() == {"status": "ok"}
    assert auth_client.get("/health/anything").status_code == 401
    assert auth_client.post("/auth/login/anything").status_code == 401
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert auth_client.get(path).status_code == 401


def test_cors_preflight_bypasses_auth_and_never_reflects_unknown_origin(
    auth_client: TestClient,
) -> None:
    headers = {
        "Origin": "http://localhost:4200",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-CSRF-Token,Content-Type",
    }
    allowed = auth_client.options("/paper/runs", headers=headers)
    denied = auth_client.options(
        "/paper/runs", headers={**headers, "Origin": "https://unknown.example"}
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:4200"
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert denied.status_code == 400
    assert "access-control-allow-origin" not in denied.headers


def test_registered_routes_do_not_expand_the_public_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_auth(monkeypatch)
    application = create_app()
    paths = application.openapi()["paths"]
    registered = {
        (method.upper(), path)
        for path, operations in paths.items()
        for method in operations
    }
    assert registered >= PUBLIC_ROUTES
    assert {
        identity
        for identity in registered
        if identity in PUBLIC_ROUTES or identity[0] == "OPTIONS"
    } == PUBLIC_ROUTES


def test_tampered_session_has_generic_unauthorized_response(
    auth_client: TestClient,
) -> None:
    auth_client.cookies.set("argos_session", "not-a-jwt")
    response = auth_client.get("/market/instruments")
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}


def test_non_admin_principal_is_forbidden_by_admin_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_auth(monkeypatch)
    application = create_app()

    @application.get("/_admin-test", dependencies=[Depends(require_admin)])
    def admin_test() -> dict[str, bool]:
        return {"ok": True}

    now = datetime.now(UTC)
    csrf = "csrf"
    token = jwt.encode(
        {
            "sub": "viewer",
            "role": "viewer",
            "iat": now,
            "exp": now + timedelta(hours=1),
            "jti": "viewer-token",
            "iss": ISSUER,
            "aud": AUDIENCE,
            "csrf": csrf_digest(csrf),
        },
        SECRET,
        algorithm=ALGORITHM,
    )
    with TestClient(application, cookies={"argos_session": token}) as client:
        assert client.get("/_admin-test").status_code == 403


def test_disabled_mode_preserves_access_docs_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    configure_auth(monkeypatch, enabled=False)
    monkeypatch.setattr("app.core.lifespan.configure_logging", lambda _level: None)
    caplog.set_level(logging.WARNING)
    with TestClient(create_app()) as client:
        assert client.get("/market/instruments").status_code == 200
        assert client.get("/docs").status_code == 200
    warnings = [
        record
        for record in caplog.records
        if "Authentication enforcement is disabled" in record.getMessage()
    ]
    assert len(warnings) == 1


def test_disabled_mode_auth_routes_work_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_auth(monkeypatch, enabled=False)
    with TestClient(create_app()) as client:
        login(client)
        assert client.get("/auth/me").status_code == 200


def test_disabled_mode_login_fails_safely_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARGOS_AUTH_ENFORCEMENT_ENABLED", "false")
    monkeypatch.delenv("ARGOS_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ARGOS_ADMIN_PASSWORD_HASH", raising=False)
    monkeypatch.delenv("ARGOS_JWT_SECRET", raising=False)
    get_settings.cache_clear()
    get_auth_settings.cache_clear()
    with TestClient(create_app()) as client:
        response = client.post(
            "/auth/login", json={"username": "admin", "password": "password"}
        )
    assert response.status_code == 503
    assert response.json() == {"detail": "Authentication is not configured"}
