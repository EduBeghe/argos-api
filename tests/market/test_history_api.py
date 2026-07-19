from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.database.session import get_db_session
from app.market.catalog import MarketCatalogAmbiguityError
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.history import HistoryOrder
from main import app
from tests.decisions.factories import make_score
from tests.decisions.test_api import snapshot


async def fake_session() -> AsyncIterator[object]:
    yield object()


def test_score_history_returns_explainable_persisted_records(
    client: TestClient,
) -> None:
    query = AsyncMock(return_value=[make_score()])
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.market.api.list_market_scores", query):
            response = client.get(
                "/market/scores",
                params={
                    "asset": "BTC/USDT",
                    "provider": "binance",
                    "market_type": "spot",
                    "interval": "5m",
                    "from": "2026-07-15T09:00:00-03:00",
                    "to": "2026-07-15T12:00:00Z",
                    "order": "asc",
                    "limit": 25,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()[0]["contributions"]["trend"] == "0.60"
    assert set(response.json()[0]["explanations"]) == {
        "trend",
        "momentum",
        "volume",
        "volatility",
    }
    query.assert_awaited_once_with(
        session=ANY,
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        from_time=datetime(2026, 7, 15, 12, tzinfo=UTC),
        to_time=datetime(2026, 7, 15, 12, tzinfo=UTC),
        order=HistoryOrder.ASC,
        limit=25,
    )


def test_decision_history_preserves_source_and_strategy_evidence(
    client: TestClient,
) -> None:
    decision = snapshot()
    query = AsyncMock(return_value=[decision])
    run_id = uuid4()
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.market.api.list_decision_snapshots", query):
            response = client.get(
                "/market/decisions",
                params={
                    "asset": "BTC/USDT",
                    "interval": "5m",
                    "run_id": str(run_id),
                },
            )
    finally:
        app.dependency_overrides.clear()

    body = response.json()[0]
    assert response.status_code == 200
    assert body["source_score_id"] == str(decision.source_score_id)
    assert body["requested_target_portfolio_percentage"] == "15.00"
    assert body["applied_rule"] == "entry_threshold_met"
    assert query.await_args is not None
    assert query.await_args.kwargs["run_id"] == run_id


@pytest.mark.parametrize("path", ["/market/scores", "/market/decisions"])
def test_valid_history_without_records_returns_empty_list(
    client: TestClient, path: str
) -> None:
    target = (
        "app.market.api.list_market_scores"
        if path.endswith("scores")
        else "app.market.api.list_decision_snapshots"
    )
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch(target, AsyncMock(return_value=[])):
            response = client.get(path, params={"asset": "ETH/USDT", "interval": "5m"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize(
    "path", ["/market/candles", "/market/scores", "/market/decisions"]
)
@pytest.mark.parametrize(
    "invalid",
    [
        {"from": "2026-07-15T12:00:00"},
        {"from": "2026-07-15T13:00:00Z", "to": "2026-07-15T12:00:00Z"},
        {"limit": 0},
        {"limit": 1001},
        {"order": "newest"},
    ],
)
def test_history_query_validation_is_consistent(
    client: TestClient, path: str, invalid: dict[str, object]
) -> None:
    response = client.get(
        path,
        params={"asset": "BTC/USDT", "interval": "5m", **invalid},
    )

    assert response.status_code == 422


def test_ambiguous_identity_fails_before_history_query(client: TestClient) -> None:
    query = AsyncMock()
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with (
            patch(
                "app.market.api.resolve_instrument",
                side_effect=MarketCatalogAmbiguityError("ambiguous identity"),
            ),
            patch("app.market.api.list_market_scores", query),
        ):
            response = client.get(
                "/market/scores", params={"asset": "BTC/USDT", "interval": "5m"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    query.assert_not_awaited()


def test_paper_trade_history_resolves_identity_and_preserves_compatibility(
    client: TestClient,
) -> None:
    run_id = uuid4()
    query = AsyncMock(return_value=[])
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with (
            patch(
                "app.paper.api.get_run",
                AsyncMock(return_value=SimpleNamespace(id=run_id)),
            ),
            patch("app.paper.api.list_trades", query),
        ):
            response = client.get(
                "/paper/trades",
                params={
                    "run_id": str(run_id),
                    "asset": "BTC/USDT",
                    "interval": "5m",
                    "from": "2026-07-15T12:00:00Z",
                    "order": "asc",
                    "limit": 1000,
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []
    assert query.await_args is not None
    called = query.await_args.kwargs
    assert called["provider"] is Provider.BINANCE
    assert called["market_type"] is MarketType.SPOT
    assert called["order"] is HistoryOrder.ASC
    assert called["limit"] == 1000
