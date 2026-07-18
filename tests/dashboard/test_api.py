from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.database.session import get_db_session
from app.dashboard.schemas import (
    DashboardOverviewResponse,
    DashboardPaperResponse,
    DashboardSystemResponse,
)
from app.paper.portfolio import PaperPortfolioValuationError
from main import app


async def fake_session() -> AsyncIterator[object]:
    yield object()


def response() -> DashboardOverviewResponse:
    return DashboardOverviewResponse(
        generated_at=datetime(2026, 7, 18, 20, tzinfo=UTC),
        system=DashboardSystemResponse(
            market_data_fresh=False,
            instruments_total=0,
            instruments_fresh=0,
        ),
        paper=DashboardPaperResponse(run=None, portfolio=None, performance=None),
        instruments=[],
    )


def test_dashboard_overview_is_exposed_and_read_only(
    client: TestClient,
) -> None:
    build = AsyncMock(return_value=response())
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.dashboard.router.build_dashboard_overview", build):
            result = client.get("/dashboard/overview")
    finally:
        app.dependency_overrides.clear()
    assert result.status_code == 200
    assert result.json()["paper"] == {
        "run": None,
        "portfolio": None,
        "performance": None,
    }
    assert "/dashboard/overview" in client.get("/openapi.json").json()["paths"]
    build.assert_awaited_once()


def test_dashboard_portfolio_evidence_failure_is_conflict(
    client: TestClient,
) -> None:
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch(
            "app.dashboard.router.build_dashboard_overview",
            AsyncMock(side_effect=PaperPortfolioValuationError("mark missing")),
        ):
            result = client.get("/dashboard/overview")
    finally:
        app.dependency_overrides.clear()
    assert result.status_code == 409
    assert result.json() == {"detail": "mark missing"}

