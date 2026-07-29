from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.auth.tokens import (
    ALGORITHM,
    AUDIENCE,
    ISSUER,
    AuthenticationError,
    create_session_token,
    csrf_digest,
    decode_session_token,
    valid_csrf,
)

SECRET = "s" * 32
NOW = datetime(2026, 7, 29, 12, tzinfo=UTC)


def payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "sub": "admin",
        "role": "admin",
        "iat": NOW,
        "exp": NOW + timedelta(hours=1),
        "jti": "identifier",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "csrf": csrf_digest("csrf"),
    }
    values.update(overrides)
    return values


def test_valid_token_round_trip_and_csrf_binding() -> None:
    token, csrf, expires_at = create_session_token(
        username="admin", secret=SECRET, lifetime_minutes=480
    )
    principal = decode_session_token(token, secret=SECRET)
    assert principal.username == "admin"
    assert principal.expires_at == expires_at
    assert valid_csrf(principal, csrf, csrf)
    assert not valid_csrf(principal, csrf, "different")


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": NOW - timedelta(seconds=1)},
        {"iss": "wrong"},
        {"aud": "wrong"},
        {"sub": ""},
        {"role": "unknown"},
        {"iat": "not-a-timestamp"},
        {"jti": ""},
        {"csrf": "not-a-digest"},
    ],
)
def test_invalid_claims_are_rejected(overrides: dict[str, object]) -> None:
    token = jwt.encode(payload(**overrides), SECRET, algorithm=ALGORITHM)
    with pytest.raises(AuthenticationError, match="invalid session"):
        decode_session_token(token, secret=SECRET)


def test_tampered_signature_unsupported_algorithm_and_missing_claim_are_rejected() -> (
    None
):
    tampered = jwt.encode(payload(), "x" * 32, algorithm=ALGORITHM)
    unsupported = jwt.encode(payload(), "s" * 48, algorithm="HS384")
    incomplete = payload()
    del incomplete["jti"]
    missing = jwt.encode(incomplete, SECRET, algorithm=ALGORITHM)
    for token in (tampered, unsupported, missing):
        with pytest.raises(AuthenticationError, match="invalid session"):
            decode_session_token(token, secret=SECRET)
