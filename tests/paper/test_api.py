from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.database.session import get_db_session
from app.market.enums import Asset
from app.paper.models import PaperRunStatus, PaperTradingRun
from app.paper.portfolio import PaperPortfolioValuationError
from main import app


async def fake_session() -> object:
    yield object()


def run(*, status: PaperRunStatus = PaperRunStatus.ACTIVE) -> PaperTradingRun:
    return PaperTradingRun(
        uuid4(),
        status,
        "USDT",
        Decimal("10000"),
        Decimal("9000"),
        1,
        Decimal("0.001"),
        Decimal("0.0005"),
        datetime(2026, 7, 17, tzinfo=UTC),
        None,
        None,
        None,
    )


def test_account_returns_active_run_and_404(client: TestClient) -> None:
    record = AsyncMock()
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with (
            patch("app.paper.api.get_active_run", AsyncMock(return_value=record)),
            patch("app.paper.api.run_domain", return_value=run()),
        ):
            response = client.get("/paper/account")
        with patch("app.paper.api.get_active_run", AsyncMock(return_value=None)):
            missing = client.get("/paper/account")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["account_currency"] == "USDT"
    assert missing.status_code == 404


def test_portfolio_selects_run_and_maps_missing_mark_to_conflict(
    client: TestClient,
) -> None:
    selected = AsyncMock()
    selected.id = uuid4()
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with (
            patch("app.paper.api.get_run", AsyncMock(return_value=selected)),
            patch(
                "app.paper.api.value_paper_run",
                AsyncMock(side_effect=PaperPortfolioValuationError("mark missing")),
            ) as valuation,
        ):
            response = client.get(f"/paper/portfolio?run_id={selected.id}")
        with patch("app.paper.api.get_run", AsyncMock(return_value=None)):
            missing = client.get(f"/paper/portfolio?run_id={uuid4()}")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 409
    assert response.json() == {"detail": "mark missing"}
    assert missing.status_code == 404
    valuation.assert_awaited_once()


def test_run_history_is_ordered_paginated_and_unknown_is_404(
    client: TestClient,
) -> None:
    active = run()
    stopped = run(status=PaperRunStatus.STOPPED)
    app.dependency_overrides[get_db_session] = fake_session
    try:
        query = AsyncMock(return_value=[active, stopped])
        with patch("app.paper.api.list_runs", query):
            response = client.get("/paper/runs?limit=2&offset=1")
        with patch("app.paper.api.get_run", AsyncMock(return_value=None)):
            missing = client.get(f"/paper/runs/{uuid4()}")
    finally:
        app.dependency_overrides.clear()
    assert [item["status"] for item in response.json()] == ["active", "stopped"]
    query.assert_awaited_once_with(ANY, limit=2, offset=1)
    assert missing.status_code == 404


def test_positions_and_trades_accept_historical_run_id(client: TestClient) -> None:
    selected = AsyncMock()
    selected.id = uuid4()
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with (
            patch("app.paper.api.get_run", AsyncMock(return_value=selected)),
            patch(
                "app.paper.api.list_positions", AsyncMock(return_value=[])
            ) as positions,
            patch("app.paper.api.list_trades", AsyncMock(return_value=[])) as trades,
        ):
            position_response = client.get(
                f"/paper/positions?run_id={selected.id}&limit=5&offset=2"
            )
            trade_response = client.get(
                f"/paper/trades?run_id={selected.id}&limit=6&offset=3"
            )
    finally:
        app.dependency_overrides.clear()
    assert position_response.status_code == trade_response.status_code == 200
    positions.assert_awaited_once()
    trades.assert_awaited_once()


def cycle() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        paper_run_id=uuid4(),
        asset="BTC/USDT",
        provider="binance",
        market_type="spot",
        interval="5m",
        strategy_name="baseline_long_only",
        strategy_version=1,
        configuration_version=1,
        scheduled_boundary=datetime(2026, 7, 17, 12, 30, tzinfo=UTC),
        source_score_id=uuid4(),
        decision_snapshot_id=uuid4(),
        paper_trade_id=None,
        result="no_action",
        started_at=datetime(2026, 7, 17, 12, 32, tzinfo=UTC),
        completed_at=datetime(2026, 7, 17, 12, 32, tzinfo=UTC),
        status="completed",
    )


def test_decision_cycle_api_filters_paginates_and_returns_404(
    client: TestClient,
) -> None:
    record = cycle()
    app.dependency_overrides[get_db_session] = fake_session
    try:
        query = AsyncMock(return_value=[record])
        with patch("app.paper.api.list_cycles", query):
            response = client.get(
                f"/paper/decision-cycles?run_id={record.paper_run_id}"
                "&asset=BTC%2FUSDT&limit=1&offset=2"
            )
        with patch("app.paper.api.get_cycle", AsyncMock(return_value=None)):
            missing = client.get(f"/paper/decision-cycles/{uuid4()}")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()[0]["result"] == "no_action"
    query.assert_awaited_once_with(
        session=ANY,
        run_id=record.paper_run_id,
        asset=Asset.BTC_USDT,
        limit=1,
        offset=2,
    )
    assert missing.status_code == 404


def equity_snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        run_id=uuid4(),
        scheduled_boundary=datetime(2026, 7, 18, 12, 30, tzinfo=UTC),
        recorded_at=datetime(2026, 7, 18, 12, 32, tzinfo=UTC),
        account_currency="USDT",
        initial_cash=Decimal("10000"),
        current_cash=Decimal("9000"),
        positions_market_value=Decimal("1200"),
        equity=Decimal("10200"),
        total_pnl=Decimal("200"),
        total_return=Decimal("0.02"),
        configuration_version=1,
        positions=[],
    )


def test_equity_snapshot_api_filters_orders_and_returns_404(
    client: TestClient,
) -> None:
    snapshot = equity_snapshot()
    selected = AsyncMock()
    selected.id = snapshot.run_id
    app.dependency_overrides[get_db_session] = fake_session
    try:
        query = AsyncMock(return_value=[snapshot])
        with (
            patch("app.paper.api.get_run", AsyncMock(return_value=selected)),
            patch("app.paper.api.list_equity_snapshots", query),
        ):
            response = client.get(
                f"/paper/equity-snapshots?run_id={snapshot.run_id}"
                "&limit=1&offset=2&order=asc"
            )
        with patch("app.paper.api.get_equity_snapshot", AsyncMock(return_value=None)):
            missing = client.get(f"/paper/equity-snapshots/{uuid4()}")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200 and response.json()[0]["equity"] == "10200"
    query.assert_awaited_once_with(
        session=ANY,
        run_id=snapshot.run_id,
        limit=1,
        offset=2,
        order="asc",
    )
    assert missing.status_code == 404
