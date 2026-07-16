from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.database.session import get_db_session
from app.market.api import build_market_status
from app.market.enums import Asset, Interval
from app.market.queries import MarketStatusAggregate
from app.market.schemas import CandleOrder, MarketCandleQuery, MarketCandleResponse
from main import app
from tests.market.factories import make_candle


async def fake_session() -> AsyncIterator[object]:
    yield object()


def test_candle_response_maps_domain_values_exactly() -> None:
    candle = make_candle()

    response = MarketCandleResponse.from_domain(candle)

    assert response.id == candle.id
    assert response.open_price == candle.open_price
    assert response.open_time.tzinfo is UTC
    assert response.asset is Asset.BTC_USDT


@pytest.mark.parametrize(
    "values",
    [
        {"asset": "unsupported", "interval": "5m"},
        {"asset": "BTC/USDT", "interval": "1m"},
        {"asset": "BTC/USDT", "interval": "5m", "order": "newest"},
        {"asset": "BTC/USDT", "interval": "5m", "limit": 0},
        {"asset": "BTC/USDT", "interval": "5m", "limit": 1001},
        {
            "asset": "BTC/USDT",
            "interval": "5m",
            "start_time": "2026-07-15T12:00:00",
        },
        {
            "asset": "BTC/USDT",
            "interval": "5m",
            "start_time": "2026-07-15T09:00:00-03:00",
        },
        {
            "asset": "BTC/USDT",
            "interval": "5m",
            "start_time": "2026-07-15T12:00:00Z",
            "end_time": "2026-07-15T11:00:00Z",
        },
    ],
)
def test_candle_query_rejects_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MarketCandleQuery.model_validate(values)


@pytest.mark.parametrize("asset", [Asset.BTC_USDT, Asset.ETH_USDT])
def test_candles_endpoint_passes_valid_filters(
    client: TestClient, asset: Asset
) -> None:
    candle = make_candle(asset=asset)
    query = AsyncMock(return_value=[candle])
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.market.api.list_market_candles", query):
            response = client.get(
                "/market/candles",
                params={
                    "asset": asset.value,
                    "interval": "5m",
                    "start_time": "2026-07-15T11:00:00Z",
                    "end_time": "2026-07-15T13:00:00Z",
                    "limit": 25,
                    "order": "asc",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()[0]["asset"] == asset.value
    assert response.json()[0]["open_price"] == "65000.10000000"
    assert query.await_args is not None
    called = query.await_args.kwargs
    assert called["asset"] is asset
    assert called["interval"] is Interval.FIVE_MINUTES
    assert called["limit"] == 25
    assert called["order"] is CandleOrder.ASC
    assert called["start_time"] == datetime(2026, 7, 15, 11, tzinfo=UTC)
    assert called["end_time"] == datetime(2026, 7, 15, 13, tzinfo=UTC)


def test_candles_endpoint_uses_defaults_and_returns_empty_result(
    client: TestClient,
) -> None:
    query = AsyncMock(return_value=[])
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.market.api.list_market_candles", query):
            response = client.get(
                "/market/candles", params={"asset": "BTC/USDT", "interval": "5m"}
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []
    assert query.await_args is not None
    assert query.await_args.kwargs["order"] is CandleOrder.DESC
    assert query.await_args.kwargs["limit"] == 100


@pytest.mark.parametrize(
    "params",
    [
        {"asset": "unsupported", "interval": "5m"},
        {"asset": "BTC/USDT", "interval": "1m"},
        {"asset": "BTC/USDT", "interval": "5m", "order": "invalid"},
        {"asset": "BTC/USDT", "interval": "5m", "limit": 0},
        {
            "asset": "BTC/USDT",
            "interval": "5m",
            "start_time": "2026-07-15T12:00:00",
        },
        {
            "asset": "BTC/USDT",
            "interval": "5m",
            "start_time": "2026-07-15T09:00:00-03:00",
        },
        {
            "asset": "BTC/USDT",
            "interval": "5m",
            "start_time": "2026-07-15T13:00:00Z",
            "end_time": "2026-07-15T12:00:00Z",
        },
    ],
)
def test_candles_endpoint_rejects_invalid_query(
    client: TestClient, params: dict[str, object]
) -> None:
    response = client.get("/market/candles", params=params)

    assert response.status_code == 422


def test_build_status_returns_supported_assets_in_order_with_freshness() -> None:
    now = datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC)
    btc = MarketStatusAggregate(
        asset=Asset.BTC_USDT,
        candle_count=4,
        latest_open_time=datetime(2026, 7, 15, 12, 0, tzinfo=UTC),
        latest_close_time=datetime(2026, 7, 15, 12, 4, 59, tzinfo=UTC),
        latest_ingested_at=datetime(2026, 7, 15, 12, 5, 5, tzinfo=UTC),
    )

    statuses = build_market_status(aggregates=[btc], now=now)

    assert [status.asset for status in statuses] == [
        Asset.BTC_USDT,
        Asset.ETH_USDT,
    ]
    assert statuses[0].candle_count == 4
    assert statuses[0].age_seconds == 151
    assert statuses[0].is_fresh is True
    assert statuses[1].candle_count == 0
    assert statuses[1].latest_open_time is None
    assert statuses[1].age_seconds is None
    assert statuses[1].is_fresh is False


def test_build_status_marks_old_latest_candle_stale() -> None:
    now = datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC)
    aggregate = MarketStatusAggregate(
        asset=Asset.BTC_USDT,
        candle_count=1,
        latest_open_time=datetime(2026, 7, 15, 11, 55, tzinfo=UTC),
        latest_close_time=datetime(2026, 7, 15, 11, 59, 59, tzinfo=UTC),
        latest_ingested_at=datetime(2026, 7, 15, 12, 0, 1, tzinfo=UTC),
    )

    status = build_market_status(aggregates=[aggregate], now=now)[0]

    assert status.is_fresh is False
    assert status.age_seconds == 451


def test_market_status_endpoint_and_openapi(client: TestClient) -> None:
    query = AsyncMock(return_value=[])
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.market.api.get_market_status_aggregates", query):
            response = client.get("/market/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert [item["asset"] for item in response.json()] == ["BTC/USDT", "ETH/USDT"]
    query.assert_awaited_once()
    paths = client.get("/openapi.json").json()["paths"]
    assert "/market/candles" in paths
    assert "/market/status" in paths
