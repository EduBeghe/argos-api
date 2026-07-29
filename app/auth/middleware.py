from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from app.auth.config import AuthConfigurationError, AuthSettings
from app.auth.tokens import (
    CSRF_COOKIE,
    CSRF_HEADER,
    SESSION_COOKIE,
    AuthenticationError,
    decode_session_token,
    valid_csrf,
)

PUBLIC_ROUTES = frozenset({("GET", "/health"), ("POST", "/auth/login")})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Enforce exact public routes and CSRF protection centrally."""

    def __init__(self, app: object, *, settings: AuthSettings) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.settings = settings

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not self.settings.enforcement_enabled:
            return await call_next(request)
        if (
            request.method == "OPTIONS"
            or (request.method, request.url.path) in PUBLIC_ROUTES
        ):
            return await call_next(request)
        try:
            secret = self.settings.require_credentials()[2]
            token = request.cookies.get(SESSION_COOKIE)
            if token is None:
                raise AuthenticationError
            principal = decode_session_token(token, secret=secret)
        except (AuthConfigurationError, AuthenticationError):
            return JSONResponse(
                status_code=401, content={"detail": "Authentication required"}
            )
        request.state.principal = principal
        if request.method not in SAFE_METHODS:
            cookie = request.cookies.get(CSRF_COOKIE)
            header = request.headers.get(CSRF_HEADER)
            if (
                cookie is None
                or header is None
                or not valid_csrf(principal, cookie, header)
            ):
                return JSONResponse(
                    status_code=403, content={"detail": "CSRF validation failed"}
                )
        return await call_next(request)
