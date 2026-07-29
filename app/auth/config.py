from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.auth.passwords import is_valid_argon2id_hash


class AuthConfigurationError(ValueError):
    """Authentication configuration is missing or unsafe."""


class AuthSettings(BaseSettings):
    """API-only authentication settings; background jobs do not load these."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    enforcement_enabled: bool = Field(
        default=False, validation_alias="ARGOS_AUTH_ENFORCEMENT_ENABLED"
    )
    admin_username: str | None = Field(
        default=None, validation_alias="ARGOS_ADMIN_USERNAME"
    )
    admin_password_hash: SecretStr | None = Field(
        default=None, validation_alias="ARGOS_ADMIN_PASSWORD_HASH"
    )
    jwt_secret: SecretStr | None = Field(
        default=None, validation_alias="ARGOS_JWT_SECRET"
    )
    session_lifetime_minutes: int = Field(
        default=480,
        ge=15,
        le=1440,
        validation_alias="ARGOS_SESSION_LIFETIME_MINUTES",
    )
    cookie_secure: bool = Field(
        default=True, validation_alias="ARGOS_AUTH_COOKIE_SECURE"
    )

    @field_validator("admin_username")
    @classmethod
    def validate_optional_username(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not 1 <= len(value) <= 128:
            raise ValueError("ARGOS_ADMIN_USERNAME must contain 1-128 characters")
        return value

    def validate_startup(self) -> None:
        """Reject incomplete credentials only when API enforcement is enabled."""

        if self.enforcement_enabled:
            self.require_credentials()

    def require_credentials(self) -> tuple[str, str, str]:
        """Return configured credentials after validating their safe structure."""

        missing: list[str] = []
        if self.admin_username is None:
            missing.append("ARGOS_ADMIN_USERNAME")
        if self.admin_password_hash is None:
            missing.append("ARGOS_ADMIN_PASSWORD_HASH")
        if self.jwt_secret is None:
            missing.append("ARGOS_JWT_SECRET")
        if missing:
            raise AuthConfigurationError(
                "missing authentication configuration: " + ", ".join(missing)
            )

        assert self.admin_username is not None
        assert self.admin_password_hash is not None
        assert self.jwt_secret is not None
        password_hash = self.admin_password_hash.get_secret_value()
        jwt_secret = self.jwt_secret.get_secret_value()
        if not is_valid_argon2id_hash(password_hash):
            raise AuthConfigurationError("ARGOS_ADMIN_PASSWORD_HASH is not Argon2id")
        if len(jwt_secret.encode("utf-8")) < 32:
            raise AuthConfigurationError(
                "ARGOS_JWT_SECRET must contain at least 32 UTF-8 bytes"
            )
        return self.admin_username, password_hash, jwt_secret


@lru_cache
def get_auth_settings() -> AuthSettings:
    return AuthSettings()
