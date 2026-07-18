from datetime import UTC, datetime
from enum import StrEnum
from typing import cast

import pytest

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.queries import MarketStatusAggregate
from app.market.status import (
    UnsupportedFreshnessIntervalError,
    build_market_statuses,
    expected_latest_closed_open_time,
)

NOW = datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC)


class OtherProvider(StrEnum):
    OTHER = "other"
    MISSING = "missing"


class OtherInterval(StrEnum):
    ONE_HOUR = "1h"


def test_statuses_use_complete_identity_without_same_asset_collision() -> None:
    other_provider = cast(Provider, OtherProvider.OTHER)
    missing_provider = cast(Provider, OtherProvider.MISSING)
    binance = Instrument(
        Asset.BTC_USDT,
        Provider.BINANCE,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    other = Instrument(
        Asset.BTC_USDT,
        other_provider,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    missing = Instrument(
        Asset.BTC_USDT,
        missing_provider,
        MarketType.SPOT,
        Interval.FIVE_MINUTES,
    )
    statuses = build_market_statuses(
        instruments=(binance, other, missing),
        aggregates=(
            _aggregate(provider=Provider.BINANCE, minute=0),
            _aggregate(provider=other_provider, minute=55),
        ),
        now=NOW,
    )

    assert statuses[binance].is_fresh is True
    assert statuses[other].is_fresh is False
    assert statuses[other].latest_open_time == datetime(2026, 7, 15, 11, 55, tzinfo=UTC)
    assert statuses[missing].candle_count == 0
    assert statuses[missing].is_fresh is False


def test_unsupported_interval_fails_explicitly() -> None:
    interval = cast(Interval, OtherInterval.ONE_HOUR)

    with pytest.raises(UnsupportedFreshnessIntervalError, match="1h"):
        expected_latest_closed_open_time(interval=interval, now=NOW)


def _aggregate(*, provider: Provider, minute: int) -> MarketStatusAggregate:
    hour = 12 if minute == 0 else 11
    open_time = datetime(2026, 7, 15, hour, minute, tzinfo=UTC)
    return MarketStatusAggregate(
        asset=Asset.BTC_USDT,
        provider=provider,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        candle_count=1,
        latest_open_time=open_time,
        latest_close_time=open_time,
        latest_ingested_at=open_time,
    )
