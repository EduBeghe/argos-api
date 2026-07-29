import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import jwt

from app.auth.contracts import AuthenticatedPrincipal

ALGORITHM = "HS256"
ISSUER = "argos-api"
AUDIENCE = "argos-web"
SESSION_COOKIE = "argos_session"
CSRF_COOKIE = "argos_csrf"
CSRF_HEADER = "X-CSRF-Token"


class AuthenticationError(ValueError):
    """A session token failed strict validation."""


def csrf_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_session_token(
    *, username: str, secret: str, lifetime_minutes: int, now: datetime | None = None
) -> tuple[str, str, datetime]:
    issued_at = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    expires_at = issued_at + timedelta(minutes=lifetime_minutes)
    csrf_value = secrets.token_urlsafe(32)
    payload = {
        "sub": username,
        "role": "admin",
        "iat": issued_at,
        "exp": expires_at,
        "jti": secrets.token_urlsafe(24),
        "iss": ISSUER,
        "aud": AUDIENCE,
        "csrf": csrf_digest(csrf_value),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM), csrf_value, expires_at


def decode_session_token(token: str, *, secret: str) -> AuthenticatedPrincipal:
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[ALGORITHM],
            issuer=ISSUER,
            audience=AUDIENCE,
            options={
                "require": ["sub", "role", "iat", "exp", "jti", "iss", "aud", "csrf"]
            },
        )
        subject = payload["sub"]
        role = payload["role"]
        csrf = payload["csrf"]
        issued_at = payload["iat"]
        expiration = payload["exp"]
        token_identifier = payload["jti"]
        if not isinstance(subject, str) or not 1 <= len(subject) <= 128:
            raise AuthenticationError
        if role not in {"admin", "viewer"}:
            raise AuthenticationError
        if (
            not isinstance(issued_at, (int, float))
            or isinstance(issued_at, bool)
            or not isinstance(expiration, (int, float))
            or isinstance(expiration, bool)
            or not isinstance(token_identifier, str)
            or not token_identifier
        ):
            raise AuthenticationError
        if (
            not isinstance(csrf, str)
            or len(csrf) != 64
            or any(character not in "0123456789abcdef" for character in csrf)
        ):
            raise AuthenticationError
        expires_at = datetime.fromtimestamp(expiration, tz=UTC)
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as error:
        raise AuthenticationError("invalid session") from error
    return AuthenticatedPrincipal(subject, role, expires_at, csrf)


def valid_csrf(principal: AuthenticatedPrincipal, cookie: str, header: str) -> bool:
    return secrets.compare_digest(cookie, header) and valid_csrf_cookie(
        principal, cookie
    )


def valid_csrf_cookie(principal: AuthenticatedPrincipal, cookie: str) -> bool:
    return secrets.compare_digest(csrf_digest(cookie), principal.csrf_digest)
