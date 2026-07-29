import pytest
from pydantic import ValidationError

from app.auth.config import AuthConfigurationError, AuthSettings
from app.core.config import Settings

VALID_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$YfDFPLLCSGMA2JVLzUaQCA$"
    "g0pQ2ia+qwLo1MkzAzaquAQB3FVqwxae6Wce+2DDjb8"
)


def configured(
    *,
    enforcement_enabled: bool = True,
    admin_username: str | None = "admin",
    admin_password_hash: str | None = VALID_HASH,
    jwt_secret: str | None = "x" * 32,
    session_lifetime_minutes: int = 480,
) -> AuthSettings:
    return AuthSettings(
        _env_file=None,
        enforcement_enabled=enforcement_enabled,
        admin_username=admin_username,
        admin_password_hash=admin_password_hash,
        jwt_secret=jwt_secret,
        session_lifetime_minutes=session_lifetime_minutes,
    )


def test_disabled_enforcement_allows_missing_secrets() -> None:
    settings = AuthSettings(_env_file=None)
    settings.validate_startup()


@pytest.mark.parametrize(
    ("settings", "variable"),
    [
        (configured(admin_username=None), "ARGOS_ADMIN_USERNAME"),
        (configured(admin_password_hash=None), "ARGOS_ADMIN_PASSWORD_HASH"),
        (configured(jwt_secret=None), "ARGOS_JWT_SECRET"),
    ],
)
def test_enabled_enforcement_requires_every_credential(
    settings: AuthSettings, variable: str
) -> None:
    with pytest.raises(AuthConfigurationError, match=variable):
        settings.validate_startup()


def test_short_secret_and_invalid_hash_are_rejected() -> None:
    with pytest.raises(AuthConfigurationError, match="32 UTF-8 bytes"):
        configured(jwt_secret="short").validate_startup()
    with pytest.raises(AuthConfigurationError, match="not Argon2id"):
        configured(admin_password_hash="not-a-hash").validate_startup()


@pytest.mark.parametrize("minutes", [15, 1440])
def test_session_duration_boundaries_are_accepted(minutes: int) -> None:
    assert (
        configured(session_lifetime_minutes=minutes).session_lifetime_minutes == minutes
    )


@pytest.mark.parametrize("minutes", [14, 1441])
def test_session_duration_outside_boundaries_is_rejected(minutes: int) -> None:
    with pytest.raises(ValidationError):
        configured(session_lifetime_minutes=minutes)


def test_shared_job_settings_do_not_require_auth_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@database/argos")
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+asyncpg://")
