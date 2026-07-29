from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.core.database.session import get_db_session
from main import app


class AvailableSession:
    async def execute(self, statement: object) -> None:
        del statement


class UnavailableSession:
    async def execute(self, statement: object) -> None:
        del statement
        raise SQLAlchemyError("database unavailable")


async def available_session() -> AsyncIterator[AvailableSession]:
    yield AvailableSession()


async def unavailable_session() -> AsyncIterator[UnavailableSession]:
    yield UnavailableSession()


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_database_health_when_available(client: TestClient) -> None:
    app.dependency_overrides[get_db_session] = available_session
    try:
        response = client.get("/health/database")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_database_health_when_unavailable(client: TestClient) -> None:
    app.dependency_overrides[get_db_session] = unavailable_session
    try:
        response = client.get("/health/database")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
