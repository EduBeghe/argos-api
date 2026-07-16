from functools import lru_cache
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = Field(
        validation_alias=AliasChoices("DATABASE_URL", "ARGOS_DATABASE_URL")
    )
    service_name: str = Field(default="argos-api", validation_alias="SERVICE_NAME")
    version: str = Field(default="0.1.0", validation_alias="APP_VERSION")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    binance_base_url: str = Field(
        default="https://api.binance.com", validation_alias="BINANCE_BASE_URL"
    )
    binance_connect_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        validation_alias="BINANCE_CONNECT_TIMEOUT_SECONDS",
    )
    binance_read_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        validation_alias="BINANCE_READ_TIMEOUT_SECONDS",
    )
    binance_ingestion_limit: int = Field(
        default=12,
        ge=1,
        le=1000,
        validation_alias="BINANCE_INGESTION_LIMIT",
    )
    cors_allowed_origins: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("http://localhost:4200",),
        validation_alias="CORS_ALLOWED_ORIGINS",
    )

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> tuple[str, ...]:
        """Parse and validate exact HTTP origins from a comma-separated value."""

        candidates = value.split(",") if isinstance(value, str) else value
        if not isinstance(candidates, (list, tuple)):
            raise TypeError("CORS_ALLOWED_ORIGINS must be a comma-separated string")

        origins: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, str):
                raise TypeError("CORS_ALLOWED_ORIGINS entries must be strings")
            origin = candidate.strip()
            if not origin:
                raise ValueError("CORS_ALLOWED_ORIGINS must not contain empty entries")
            if origin == "*":
                raise ValueError("CORS_ALLOWED_ORIGINS must not contain a wildcard")
            parsed = urlsplit(origin)
            try:
                parsed_port = parsed.port
            except ValueError as exc:
                raise ValueError(
                    "CORS_ALLOWED_ORIGINS contains an invalid port"
                ) from exc
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
                or (":" in parsed.netloc and parsed_port is None)
            ):
                raise ValueError(
                    "CORS_ALLOWED_ORIGINS entries must be exact HTTP origins"
                )
            origins.append(origin)
        return tuple(origins)

    @field_validator("database_url", mode="before")
    @classmethod
    def use_asyncpg_driver(cls, value: object) -> str:
        """Normalize standard PostgreSQL URLs for SQLAlchemy asyncpg usage."""

        if not isinstance(value, str):
            raise TypeError("DATABASE_URL must be a string")
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+asyncpg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the cached process-wide settings."""

    return Settings()
