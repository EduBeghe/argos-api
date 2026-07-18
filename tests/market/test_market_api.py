from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import ANY, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.database.session import get_db_session
from app.features.models import FeatureName, MarketFeature
from app.market.api import build_market_status
from app.market.catalog import Instrument, MarketCatalogError
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.queries import MarketStatusAggregate
from app.market.schemas import CandleOrder, MarketCandleQuery, MarketCandleResponse
from app.scores.calculator import ScoreInterpretation
from app.scores.models import MarketScore
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
    assert called["provider"] is Provider.BINANCE
    assert called["market_type"] is MarketType.SPOT
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
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
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
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
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
        with patch("app.market.api.get_market_status_aggregates_batch", query):
            response = client.get("/market/status")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert [item["asset"] for item in response.json()] == ["BTC/USDT", "ETH/USDT"]
    query.assert_awaited_once()
    assert query.await_args is not None
    assert query.await_args.kwargs["instruments"] == (
        Instrument(
            Asset.BTC_USDT,
            Provider.BINANCE,
            MarketType.SPOT,
            Interval.FIVE_MINUTES,
        ),
        Instrument(
            Asset.ETH_USDT,
            Provider.BINANCE,
            MarketType.SPOT,
            Interval.FIVE_MINUTES,
        ),
    )
    paths = client.get("/openapi.json").json()["paths"]
    assert "/market/candles" in paths
    assert "/market/status" in paths


def test_instruments_endpoint_is_static_ordered_and_complete(
    client: TestClient,
) -> None:
    response = client.get("/market/instruments")
    assert response.status_code == 200
    payload = response.json()
    assert [item["asset"] for item in payload] == ["BTC/USDT", "ETH/USDT"]
    assert [item["display_order"] for item in payload] == [1, 2]
    assert payload[0] == {
        "asset": "BTC/USDT",
        "display_symbol": "BTC/USDT",
        "display_name": "Bitcoin / Tether",
        "asset_class": "crypto",
        "base_currency": "BTC",
        "quote_currency": "USDT",
        "provider": "binance",
        "provider_symbol": "BTCUSDT",
        "market_type": "spot",
        "interval": "5m",
        "capabilities": {
            "ingestion": True,
            "scoring": True,
            "decision": True,
            "paper_trading": True,
        },
        "display_order": 1,
    }


def test_market_status_uses_only_binance_ingestion_projection(
    client: TestClient,
) -> None:
    btc = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    query = AsyncMock(return_value=[])
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with (
            patch(
                "app.market.api.binance_spot_five_minute_ingestion_instruments",
                return_value=(btc,),
            ),
            patch("app.market.api.get_market_status_aggregates_batch", query),
        ):
            response = client.get("/market/status")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert [item["asset"] for item in response.json()] == ["BTC/USDT"]
    query.assert_awaited_once_with(
        session=ANY,
        instruments=(btc,),
    )


@pytest.mark.parametrize(
    "path",
    [
        "/market/candles?asset=BTC%2FUSDT&interval=5m",
        "/market/features/latest?asset=BTC%2FUSDT&interval=5m",
        "/market/scores/latest?asset=BTC%2FUSDT&interval=5m",
        "/market/decisions/latest?asset=BTC%2FUSDT&interval=5m",
    ],
)
def test_unconfigured_instrument_fails_before_database_query(
    client: TestClient, path: str
) -> None:
    queries = {
        "app.market.api.list_market_candles": AsyncMock(),
        "app.market.api.get_latest_feature_snapshot": AsyncMock(),
        "app.market.api.get_latest_market_score": AsyncMock(),
        "app.market.api.get_latest_decision_snapshot": AsyncMock(),
    }
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with (
            patch(
                "app.market.api.resolve_instrument",
                side_effect=MarketCatalogError("instrument not configured"),
            ),
            patch(next(iter(queries)), next(iter(queries.values()))),
            patch(list(queries)[1], list(queries.values())[1]),
            patch(list(queries)[2], list(queries.values())[2]),
            patch(list(queries)[3], list(queries.values())[3]),
        ):
            response = client.get(path)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 404
    for query in queries.values():
        query.assert_not_awaited()


def test_latest_feature_endpoint_returns_exact_ema(client: TestClient) -> None:
    feature_time = datetime(2026, 7, 15, 12, tzinfo=UTC)
    feature = MarketFeature(
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        candle_open_time=feature_time,
        feature_name=FeatureName.EMA_20,
        feature_version=1,
        value=Decimal("65000.123456789012345678"),
        calculated_at=feature_time,
    )
    rsi = MarketFeature(
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        candle_open_time=feature_time,
        feature_name=FeatureName.RSI_14,
        feature_version=1,
        value=Decimal("54.123456789012345678"),
        calculated_at=feature_time,
    )
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch(
            "app.market.api.get_latest_feature_snapshot",
            AsyncMock(return_value=[feature, rsi]),
        ):
            response = client.get(
                "/market/features/latest",
                params={"asset": "BTC/USDT", "interval": "5m"},
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["features"]["ema_20"] == "65000.123456789012345678"
    assert response.json()["features"]["rsi_14"] == "54.123456789012345678"
    assert response.json()["features"]["ema_50"] is None
    assert response.json()["candle_open_time"] == "2026-07-15T12:00:00Z"


def test_latest_feature_endpoint_returns_404_without_calculation(
    client: TestClient,
) -> None:
    query = AsyncMock(return_value=[])
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch("app.market.api.get_latest_feature_snapshot", query):
            response = client.get(
                "/market/features/latest",
                params={"asset": "ETH/USDT", "interval": "5m"},
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 404
    query.assert_awaited_once()
    assert "/market/features/latest" in client.get("/openapi.json").json()["paths"]


def score(asset: Asset = Asset.BTC_USDT) -> MarketScore:
    return MarketScore(
        asset=asset,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        candle_open_time=datetime(2026, 7, 15, 12, tzinfo=UTC),
        model_name="baseline_technical",
        model_version=1,
        calculated_at=datetime(2026, 7, 16, tzinfo=UTC),
        trend_score=Decimal("0.5"),
        momentum_score=Decimal("0.25"),
        volume_activity=Decimal("0.333333333333333333"),
        volatility_ratio=Decimal("0.002"),
        directional_score=Decimal("0.3875"),
        volume_confirmation_factor=Decimal("0.83333333333333333325"),
        overall_score=Decimal("0.322916666666666666634375"),
        signal_strength=Decimal("0.322916666666666666634375"),
        trend_contribution=Decimal("0.275"),
        rsi_contribution=Decimal("0.05"),
        return_1_contribution=Decimal("0.02"),
        return_12_contribution=Decimal("0.0425"),
        interpretation=ScoreInterpretation.BULLISH,
    )


def test_latest_score_returns_exact_explainable_snapshot(client: TestClient) -> None:
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch(
            "app.market.api.get_latest_market_score", AsyncMock(return_value=score())
        ):
            response = client.get(
                "/market/scores/latest", params={"asset": "BTC/USDT", "interval": "5m"}
            )
    finally:
        app.dependency_overrides.clear()
    body = response.json()
    assert response.status_code == 200
    assert body["overall_score"] == "0.322916666666666666634375"
    assert body["contributions"]["trend"] == "0.275"
    assert body["interpretation"] == "bullish"
    assert set(body["explanations"]) == {"trend", "momentum", "volume", "volatility"}
    assert "action" not in body
    assert "/market/scores/latest" in client.get("/openapi.json").json()["paths"]


def test_latest_score_returns_404_without_persisted_score(client: TestClient) -> None:
    app.dependency_overrides[get_db_session] = fake_session
    try:
        with patch(
            "app.market.api.get_latest_market_score", AsyncMock(return_value=None)
        ):
            response = client.get(
                "/market/scores/latest", params={"asset": "ETH/USDT", "interval": "5m"}
            )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 404
