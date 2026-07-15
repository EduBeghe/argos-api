from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

import pytest

from app.ingestion.binance.normalization_errors import (
    BinanceKlineDomainError,
    BinanceKlineFieldError,
    BinanceKlineStructureError,
)
from app.ingestion.binance.normalizer import normalize_kline
from app.ingestion.binance.types import BinanceKline
from app.market.enums import Asset, Interval, MarketType, Provider

OPEN_TIME_MS = 1_700_000_000_000
CLOSE_TIME_MS = OPEN_TIME_MS + 299_999
INGESTED_AT = datetime(2023, 11, 14, 22, 18, 21, tzinfo=UTC)


def raw_kline() -> BinanceKline:
    """Return one valid raw Binance five-minute kline."""

    return [
        OPEN_TIME_MS,
        "35000.12345678",
        "35100.00000000",
        "34900.10000000",
        "35050.87654321",
        "12.34567890",
        CLOSE_TIME_MS,
        "432100.12345678",
        987,
        "6.12345678",
        "214000.87654321",
        "0",
    ]


def test_normalize_kline_maps_every_retained_field_exactly() -> None:
    candle = normalize_kline(
        raw_kline=raw_kline(),
        asset=Asset.BTC_USDT,
        interval=Interval.FIVE_MINUTES,
        ingested_at=INGESTED_AT,
    )

    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    assert candle.id.version == 4
    assert candle.asset is Asset.BTC_USDT
    assert candle.provider is Provider.BINANCE
    assert candle.market_type is MarketType.SPOT
    assert candle.interval is Interval.FIVE_MINUTES
    assert candle.open_time == epoch + timedelta(milliseconds=OPEN_TIME_MS)
    assert candle.close_time == epoch + timedelta(milliseconds=CLOSE_TIME_MS)
    assert candle.open_price == Decimal("35000.12345678")
    assert candle.high_price == Decimal("35100.00000000")
    assert candle.low_price == Decimal("34900.10000000")
    assert candle.close_price == Decimal("35050.87654321")
    assert candle.base_asset_volume == Decimal("12.34567890")
    assert candle.quote_asset_volume == Decimal("432100.12345678")
    assert candle.trade_count == 987
    assert candle.taker_buy_base_asset_volume == Decimal("6.12345678")
    assert candle.taker_buy_quote_asset_volume == Decimal("214000.87654321")
    assert candle.ingested_at == INGESTED_AT
    assert not hasattr(candle, "ignore")


@pytest.mark.parametrize("length", [0, 11, 13])
def test_normalize_kline_rejects_wrong_position_count(length: int) -> None:
    row = raw_kline()
    if length > len(row):
        row.extend([None] * (length - len(row)))
    else:
        row = row[:length]

    with pytest.raises(BinanceKlineStructureError, match="exactly 12"):
        normalize_kline(
            raw_kline=row,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )


def test_normalize_kline_rejects_non_list_structure() -> None:
    with pytest.raises(BinanceKlineStructureError):
        normalize_kline(
            raw_kline=cast(BinanceKline, ()),
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )


@pytest.mark.parametrize(
    ("index", "invalid_value", "field"),
    [
        (0, str(OPEN_TIME_MS), "open time"),
        (1, 35000, "open price"),
        (6, str(CLOSE_TIME_MS), "close time"),
        (8, "987", "trade count"),
        (8, True, "trade count"),
        (11, 0, "ignore"),
    ],
)
def test_normalize_kline_rejects_wrong_provider_field_types(
    index: int, invalid_value: object, field: str
) -> None:
    row = raw_kline()
    row[index] = invalid_value

    with pytest.raises(BinanceKlineFieldError, match=field) as captured:
        normalize_kline(
            raw_kline=row,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )

    assert captured.value.index == index
    assert captured.value.field == field


@pytest.mark.parametrize("value", ["not-decimal", "NaN", "Infinity"])
def test_normalize_kline_rejects_invalid_decimal_strings(value: str) -> None:
    row = raw_kline()
    row[1] = value

    with pytest.raises(BinanceKlineFieldError, match="open price"):
        normalize_kline(
            raw_kline=row,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )


@pytest.mark.parametrize("timestamp", [-1, 10**30])
def test_normalize_kline_rejects_invalid_timestamps(timestamp: int) -> None:
    row = raw_kline()
    row[0] = timestamp

    with pytest.raises(BinanceKlineFieldError, match="open time"):
        normalize_kline(
            raw_kline=row,
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )


def test_normalize_kline_wraps_market_candle_validation() -> None:
    row = raw_kline()
    row[2] = "34000"

    with pytest.raises(BinanceKlineDomainError) as captured:
        normalize_kline(
            raw_kline=row,
            asset=Asset.ETH_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=INGESTED_AT,
        )

    assert captured.value.__cause__ is not None


def test_normalize_kline_rejects_naive_ingestion_time() -> None:
    with pytest.raises(BinanceKlineDomainError):
        normalize_kline(
            raw_kline=raw_kline(),
            asset=Asset.BTC_USDT,
            interval=Interval.FIVE_MINUTES,
            ingested_at=datetime(2023, 11, 14, 22, 18, 21),
        )
