from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request

from app.auth.config import AuthConfigurationError, AuthSettings, get_auth_settings
from app.auth.contracts import AuthenticatedPrincipal
from app.auth.tokens import (
    CSRF_COOKIE,
    CSRF_HEADER,
    SESSION_COOKIE,
    AuthenticationError,
    decode_session_token,
    valid_csrf,
)

AuthSettingsDependency = Annotated[AuthSettings, Depends(get_auth_settings)]


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=401, detail="Authentication required")


def _configured_secret(settings: AuthSettings) -> str:
    try:
        return settings.require_credentials()[2]
    except AuthConfigurationError as error:
        raise HTTPException(
            status_code=503, detail="Authentication is not configured"
        ) from error


def require_authenticated(
    request: Request, settings: AuthSettingsDependency
) -> AuthenticatedPrincipal:
    existing = getattr(request.state, "principal", None)
    if isinstance(existing, AuthenticatedPrincipal):
        return existing
    token = request.cookies.get(SESSION_COOKIE)
    if token is None:
        raise _unauthorized()
    try:
        principal = decode_session_token(token, secret=_configured_secret(settings))
    except AuthenticationError as error:
        raise _unauthorized() from error
    request.state.principal = principal
    return principal


PrincipalDependency = Annotated[AuthenticatedPrincipal, Depends(require_authenticated)]


def require_admin(principal: PrincipalDependency) -> AuthenticatedPrincipal:
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access required")
    return principal


def require_csrf(request: Request, principal: PrincipalDependency) -> None:
    cookie = request.cookies.get(CSRF_COOKIE)
    header = request.headers.get(CSRF_HEADER)
    if cookie is None or header is None or not valid_csrf(principal, cookie, header):
        raise HTTPException(status_code=403, detail="CSRF validation failed")


def request_principal(request: Request) -> AuthenticatedPrincipal:
    return cast(AuthenticatedPrincipal, request.state.principal)
