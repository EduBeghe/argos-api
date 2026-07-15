from pytest import MonkeyPatch

from app.core.config import Settings


def test_settings_load_from_environment(monkeypatch: MonkeyPatch) -> None:
    environment = {
        "DATABASE_URL": "postgresql+asyncpg://user:secret@database:5432/argos",
        "SERVICE_NAME": "test-api",
        "APP_VERSION": "9.9.9",
        "LOG_LEVEL": "DEBUG",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    settings = Settings(_env_file=None)

    assert settings.database_url == environment["DATABASE_URL"]
    assert settings.service_name == "test-api"
    assert settings.version == "9.9.9"
    assert settings.log_level == "DEBUG"


def test_settings_normalize_railway_database_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:secret@database:5432/argos")

    settings = Settings(_env_file=None)

    assert settings.database_url == (
        "postgresql+asyncpg://user:secret@database:5432/argos"
    )
