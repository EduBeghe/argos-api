from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.features.indicators import (
    CalculationPoint,
    calculate_atr,
    calculate_ema,
    calculate_relative_volume,
    calculate_return,
    calculate_rsi,
)
from app.optimization.historical_data import HistoricalDataset


@dataclass(frozen=True, slots=True)
class HistoricalFeatureRow:
    candle_open_time: datetime
    close_price: Decimal
    ema_20: Decimal
    ema_50: Decimal
    rsi_14: Decimal
    atr_14: Decimal
    return_1: Decimal
    return_12: Decimal
    relative_volume_20: Decimal


@dataclass(frozen=True, slots=True)
class HistoricalFeatureMatrix:
    dataset_fingerprint: str
    rows: tuple[HistoricalFeatureRow, ...]

    @property
    def first_feature_time(self) -> datetime | None:
        return self.rows[0].candle_open_time if self.rows else None

    @property
    def last_feature_time(self) -> datetime | None:
        return self.rows[-1].candle_open_time if self.rows else None


def build_historical_feature_matrix(
    dataset: HistoricalDataset,
) -> HistoricalFeatureMatrix:
    """Calculate the seven version-1 Features once, entirely in memory."""

    candles = dataset.candles
    closes = [candle.close_price for candle in candles]
    highs = [candle.high_price for candle in candles]
    lows = [candle.low_price for candle in candles]
    volumes = [candle.quote_asset_volume for candle in candles]
    calculated = (
        calculate_ema(closes, period=20),
        calculate_ema(closes, period=50),
        calculate_rsi(closes, period=14),
        calculate_atr(highs, lows, closes, period=14),
        calculate_return(closes, period=1),
        calculate_return(closes, period=12),
        calculate_relative_volume(volumes, period=20),
    )
    indexed = tuple(_by_index(points) for points in calculated)
    eligible = sorted(set.intersection(*(set(values) for values in indexed)))
    rows = tuple(
        HistoricalFeatureRow(
            candle_open_time=candles[index].open_time,
            close_price=candles[index].close_price,
            ema_20=indexed[0][index],
            ema_50=indexed[1][index],
            rsi_14=indexed[2][index],
            atr_14=indexed[3][index],
            return_1=indexed[4][index],
            return_12=indexed[5][index],
            relative_volume_20=indexed[6][index],
        )
        for index in eligible
    )
    return HistoricalFeatureMatrix(dataset.fingerprint, rows)


def _by_index(points: list[CalculationPoint]) -> dict[int, Decimal]:
    return {point.candle_index: point.value for point in points}
