from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.market.persistence import MarketCandleRecord
from app.ml.registry import MAXIMUM_HISTORICAL_LOOKBACK
from app.optimization.historical_data import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

from .errors import MLDatasetError
from .gaps import GapSummary, datetime_to_ns, summarize_gaps


@dataclass(frozen=True, slots=True)
class MLCandleColumns:
    """Compact, ordered candle columns for exactly one instrument."""

    open_time_ns: NDArray[np.int64]
    close_time_ns: NDArray[np.int64]
    open_price: NDArray[np.float64]
    high_price: NDArray[np.float64]
    low_price: NDArray[np.float64]
    close_price: NDArray[np.float64]
    base_asset_volume: NDArray[np.float64]

    def __len__(self) -> int:
        return len(self.open_time_ns)


@dataclass(frozen=True, slots=True)
class MLCandleReadResult:
    columns: MLCandleColumns
    gaps: GapSummary


async def read_live_ml_candles(
    *, session: AsyncSession, instrument: Instrument, boundary: datetime
) -> MLCandleColumns:
    """Read only the bounded consecutive context needed for one live prediction."""

    maximum_steps = int(MAXIMUM_HISTORICAL_LOOKBACK.total_seconds() // 300)
    rows = list(
        (
            await session.execute(
                select(
                    MarketCandleRecord.open_time,
                    MarketCandleRecord.close_time,
                    MarketCandleRecord.open_price,
                    MarketCandleRecord.high_price,
                    MarketCandleRecord.low_price,
                    MarketCandleRecord.close_price,
                    MarketCandleRecord.base_asset_volume,
                )
                .where(
                    MarketCandleRecord.asset == instrument.asset.value,
                    MarketCandleRecord.provider == instrument.provider.value,
                    MarketCandleRecord.market_type == instrument.market_type.value,
                    MarketCandleRecord.interval == instrument.interval.value,
                    MarketCandleRecord.close_time <= boundary,
                )
                .order_by(MarketCandleRecord.open_time.desc())
                .limit(maximum_steps)
            )
        )
        .tuples()
        .all()
    )
    rows.reverse()
    return candle_columns(
        open_times=[row[0] for row in rows],
        close_times=[row[1] for row in rows],
        open_prices=[float(row[2]) for row in rows],
        high_prices=[float(row[3]) for row in rows],
        low_prices=[float(row[4]) for row in rows],
        close_prices=[float(row[5]) for row in rows],
        volumes=[float(row[6]) for row in rows],
    )


def candle_columns(
    *,
    open_times: list[datetime],
    close_times: list[datetime],
    open_prices: list[float],
    high_prices: list[float],
    low_prices: list[float],
    close_prices: list[float],
    volumes: list[float],
) -> MLCandleColumns:
    if any(value.tzinfo is not UTC for value in (*open_times, *close_times)):
        raise MLDatasetError("all candle timestamps must be timezone-aware UTC")
    columns = MLCandleColumns(
        np.asarray([datetime_to_ns(value) for value in open_times], dtype=np.int64),
        np.asarray([datetime_to_ns(value) for value in close_times], dtype=np.int64),
        np.asarray(open_prices, dtype=np.float64),
        np.asarray(high_prices, dtype=np.float64),
        np.asarray(low_prices, dtype=np.float64),
        np.asarray(close_prices, dtype=np.float64),
        np.asarray(volumes, dtype=np.float64),
    )
    lengths = {
        len(columns.open_time_ns),
        len(columns.close_time_ns),
        len(columns.open_price),
        len(columns.high_price),
        len(columns.low_price),
        len(columns.close_price),
        len(columns.base_asset_volume),
    }
    if len(lengths) != 1:
        raise MLDatasetError("candle columns must have equal lengths")
    return columns


async def read_ml_experiment_candles(
    *,
    session: AsyncSession,
    instrument: Instrument,
    start: datetime,
    end: datetime,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> MLCandleReadResult:
    """Read one instrument into compact columns using bounded keyset pages."""

    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise MLDatasetError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
    if start.tzinfo is not UTC or end.tzinfo is not UTC or start >= end:
        raise MLDatasetError("ML candle bounds must be valid UTC timestamps")
    opens_at: list[datetime] = []
    closes_at: list[datetime] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[float] = []
    cursor: datetime | None = None
    while True:
        statement = select(
            MarketCandleRecord.open_time,
            MarketCandleRecord.close_time,
            MarketCandleRecord.open_price,
            MarketCandleRecord.high_price,
            MarketCandleRecord.low_price,
            MarketCandleRecord.close_price,
            MarketCandleRecord.base_asset_volume,
        ).where(
            MarketCandleRecord.asset == instrument.asset.value,
            MarketCandleRecord.provider == instrument.provider.value,
            MarketCandleRecord.market_type == instrument.market_type.value,
            MarketCandleRecord.interval == instrument.interval.value,
            MarketCandleRecord.open_time >= start,
            MarketCandleRecord.open_time < end,
        )
        if cursor is not None:
            statement = statement.where(MarketCandleRecord.open_time > cursor)
        rows = (
            (
                await session.execute(
                    statement.order_by(MarketCandleRecord.open_time).limit(page_size)
                )
            )
            .tuples()
            .all()
        )
        if not rows:
            break
        for row in rows:
            opens_at.append(row[0])
            closes_at.append(row[1])
            opens.append(float(row[2]))
            highs.append(float(row[3]))
            lows.append(float(row[4]))
            closes.append(float(row[5]))
            volumes.append(float(row[6]))
        next_cursor = rows[-1][0]
        if cursor is not None and next_cursor <= cursor:
            raise MLDatasetError("ML candle keyset cursor did not advance")
        cursor = next_cursor
        if len(rows) < page_size:
            break
    columns = candle_columns(
        open_times=opens_at,
        close_times=closes_at,
        open_prices=opens,
        high_prices=highs,
        low_prices=lows,
        close_prices=closes,
        volumes=volumes,
    )
    return MLCandleReadResult(columns, summarize_gaps(columns.open_time_ns))
