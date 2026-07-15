import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://argos:argos@localhost:5432/argos"
)

from app.core.config import get_settings
from main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide a test client with application lifespan enabled."""

    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()
