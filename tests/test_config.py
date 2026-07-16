import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from app.core.config import Settings


def test_settings_load_from_environment(monkeypatch: MonkeyPatch) -> None:
    environment = {
        "DATABASE_URL": "postgresql+asyncpg://user:secret@database:5432/argos",
        "SERVICE_NAME": "test-api",
        "APP_VERSION": "9.9.9",
        "LOG_LEVEL": "DEBUG",
        "BINANCE_BASE_URL": "https://binance.test",
        "BINANCE_CONNECT_TIMEOUT_SECONDS": "3",
        "BINANCE_READ_TIMEOUT_SECONDS": "8",
        "BINANCE_INGESTION_LIMIT": "24",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.database_url == environment["DATABASE_URL"]
    assert settings.service_name == "test-api"
    assert settings.version == "9.9.9"
    assert settings.log_level == "DEBUG"
    assert settings.binance_base_url == "https://binance.test"
    assert settings.binance_connect_timeout_seconds == 3.0
    assert settings.binance_read_timeout_seconds == 8.0
    assert settings.binance_ingestion_limit == 24


def test_binance_settings_have_conservative_defaults(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@database/argos")

    settings = Settings(_env_file=None)

    assert settings.binance_base_url == "https://api.binance.com"
    assert settings.binance_connect_timeout_seconds == 5.0
    assert settings.binance_read_timeout_seconds == 10.0
    assert settings.binance_ingestion_limit == 12


def test_settings_normalize_railway_database_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@database:5432/argos")

    settings = Settings(_env_file=None)

    assert settings.database_url == (
        "postgresql+asyncpg://user:secret@database:5432/argos"
    )


@pytest.mark.parametrize("limit", [0, 1001])
def test_binance_ingestion_limit_is_bounded(
    monkeypatch: MonkeyPatch, limit: int
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@database/argos")
    monkeypatch.setenv("BINANCE_INGESTION_LIMIT", str(limit))

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
