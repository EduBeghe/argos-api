from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.features.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_relative_volume,
    calculate_return,
    calculate_rsi,
)
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import (
    HistoricalCandle,
    HistoricalDataset,
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
)

START = datetime(2024, 1, 1, tzinfo=UTC)


def dataset(count: int = 60) -> HistoricalDataset:
    instrument = Instrument(
        Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
    )
    candles = tuple(
        HistoricalCandle(
            opened := START + timedelta(minutes=5 * index),
            opened + timedelta(minutes=5, milliseconds=-1),
            Decimal(index + 1),
            Decimal(index + 3),
            Decimal(index),
            Decimal(index + 2),
            Decimal(1),
            Decimal(100 + index),
            index,
            Decimal("0.5"),
            Decimal(50),
        )
        for index in range(count)
    )
    identity = HistoricalDatasetIdentity(
        instrument, HistoricalDatasetRange(START, START + timedelta(minutes=5 * count))
    )
    return HistoricalDataset(
        identity,
        count,
        count,
        candles[0].open_time,
        candles[-1].open_time,
        0,
        None,
        None,
        candles,
        "fingerprint",
    )


def test_matrix_is_ordered_excludes_warmup_and_matches_all_pure_features() -> None:
    source = dataset()
    matrix = build_historical_feature_matrix(source)
    closes = [item.close_price for item in source.candles]
    highs = [item.high_price for item in source.candles]
    lows = [item.low_price for item in source.candles]
    volumes = [item.quote_asset_volume for item in source.candles]
    expected = [
        calculate_ema(closes, period=20),
        calculate_ema(closes, period=50),
        calculate_rsi(closes, period=14),
        calculate_atr(highs, lows, closes, period=14),
        calculate_return(closes, period=1),
        calculate_return(closes, period=12),
        calculate_relative_volume(volumes, period=20),
    ]
    maps = [
        {point.candle_index: point.value for point in values} for values in expected
    ]
    assert matrix.rows[0].candle_open_time == source.candles[49].open_time
    assert [row.candle_open_time for row in matrix.rows] == sorted(
        row.candle_open_time for row in matrix.rows
    )
    for offset, row in enumerate(matrix.rows, start=49):
        assert (
            row.ema_20,
            row.ema_50,
            row.rsi_14,
            row.atr_14,
            row.return_1,
            row.return_12,
            row.relative_volume_20,
        ) == tuple(values[offset] for values in maps)
    assert matrix.dataset_fingerprint == source.fingerprint
