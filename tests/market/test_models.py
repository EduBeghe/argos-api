from datetime import datetime
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.models import MarketCandle
from tests.market.factories import candle_data, make_candle


def test_market_candle_accepts_valid_exact_data() -> None:
    candle = make_candle()

    assert isinstance(candle.id, UUID)
    assert candle.id.version == 4
    assert candle.open_price == Decimal("65000.10000000")
    assert candle.asset is Asset.BTC_USDT


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("open_price", Decimal("0")),
        ("base_asset_volume", Decimal("-0.1")),
        ("trade_count", -1),
        ("high_price", Decimal("64000")),
        ("low_price", Decimal("66000")),
    ],
)
def test_market_candle_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MarketCandle.model_validate(candle_data(**{field: value}))


def test_market_candle_rejects_reversed_times() -> None:
    with pytest.raises(ValidationError, match="close_time"):
        MarketCandle.model_validate(
            candle_data(close_time=datetime.fromisoformat("2026-07-15T11:59:59+00:00"))
        )


@pytest.mark.parametrize("field", ["open_time", "close_time", "ingested_at"])
def test_market_candle_rejects_naive_datetimes(field: str) -> None:
    with pytest.raises(ValidationError):
        MarketCandle.model_validate(
            candle_data(**{field: datetime.fromisoformat("2026-07-15T12:00:00")})
        )


def test_market_enums_use_stable_external_values() -> None:
    assert Asset.BTC_USDT.value == "BTC/USDT"
    assert Asset.ETH_USDT.value == "ETH/USDT"
    assert Provider.BINANCE.value == "binance"
    assert MarketType.SPOT.value == "spot"
    assert Interval.FIVE_MINUTES.value == "5m"
