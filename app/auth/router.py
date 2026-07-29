from fastapi import APIRouter, Depends, HTTPException, Response

from app.auth.config import AuthConfigurationError
from app.auth.contracts import LoginRequest, SessionResponse
from app.auth.dependencies import (
    AuthSettingsDependency,
    BoundCsrfCookieDependency,
    PrincipalDependency,
    require_csrf,
)
from app.auth.passwords import verify_admin_credentials
from app.auth.tokens import (
    CSRF_COOKIE,
    CSRF_HEADER,
    SESSION_COOKIE,
    create_session_token,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


def _cookie_options(*, secure: bool) -> dict[str, object]:
    return {
        "secure": secure,
        "samesite": "none" if secure else "lax",
        "path": "/",
    }


@router.post("/login", status_code=204)
def login(
    credentials: LoginRequest, response: Response, settings: AuthSettingsDependency
) -> None:
    try:
        username, password_hash, secret = settings.require_credentials()
    except AuthConfigurationError as error:
        raise HTTPException(
            status_code=503, detail="Authentication is not configured"
        ) from error
    if not verify_admin_credentials(
        submitted_username=credentials.username,
        submitted_password=credentials.password,
        configured_username=username,
        configured_password_hash=password_hash,
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token, csrf, _ = create_session_token(
        username=username,
        secret=secret,
        lifetime_minutes=settings.session_lifetime_minutes,
    )
    options = _cookie_options(secure=settings.cookie_secure)
    response.set_cookie(SESSION_COOKIE, token, httponly=True, **options)  # type: ignore[arg-type]
    response.set_cookie(CSRF_COOKIE, csrf, httponly=False, **options)  # type: ignore[arg-type]
    response.headers[CSRF_HEADER] = csrf
    response.headers["Cache-Control"] = "no-store"


@router.get("/me", response_model=SessionResponse)
def me(
    response: Response,
    principal: PrincipalDependency,
    csrf: BoundCsrfCookieDependency,
) -> SessionResponse:
    response.headers[CSRF_HEADER] = csrf
    response.headers["Cache-Control"] = "no-store"
    return SessionResponse(
        username=principal.username,
        role="admin",
        expires_at=principal.expires_at,
    )


@router.post("/logout", status_code=204, dependencies=[Depends(require_csrf)])
def logout(response: Response, settings: AuthSettingsDependency) -> None:
    options = _cookie_options(secure=settings.cookie_secure)
    response.delete_cookie(SESSION_COOKIE, httponly=True, **options)  # type: ignore[arg-type]
    response.delete_cookie(CSRF_COOKIE, httponly=False, **options)  # type: ignore[arg-type]
    response.headers["Cache-Control"] = "no-store"
