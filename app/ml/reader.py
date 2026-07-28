from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.market.persistence import MarketCandleRecord
from app.optimization.historical_data import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    HistoricalCandle,
)

from .contract import CANDLE_INTERVAL
from .dataset import MLDatasetError


@dataclass(frozen=True, slots=True)
class MLCandleReadResult:
    candles: tuple[HistoricalCandle, ...]
    first_open_time: datetime | None
    last_open_time: datetime | None
    detected_gap_count: int


async def read_ml_experiment_candles(
    *,
    session: AsyncSession,
    instrument: Instrument,
    start: datetime,
    end: datetime,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> MLCandleReadResult:
    """Read one instrument with bounded keyset pages without rejecting genuine gaps."""

    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise MLDatasetError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise MLDatasetError("ML candle bounds must be valid timezone-aware timestamps")
    candles: list[HistoricalCandle] = []
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
            MarketCandleRecord.quote_asset_volume,
            MarketCandleRecord.trade_count,
            MarketCandleRecord.taker_buy_base_asset_volume,
            MarketCandleRecord.taker_buy_quote_asset_volume,
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
                    statement.order_by(MarketCandleRecord.open_time.asc()).limit(
                        page_size
                    )
                )
            )
            .tuples()
            .all()
        )
        if not rows:
            break
        page = tuple(HistoricalCandle(*row) for row in rows)
        if cursor is not None and page[0].open_time <= cursor:
            raise MLDatasetError("ML candle keyset cursor did not advance")
        candles.extend(page)
        cursor = page[-1].open_time
        if len(page) < page_size:
            break
    result = tuple(candles)
    gaps = sum(
        1
        for left, right in pairwise(result)
        if right.open_time - left.open_time != CANDLE_INTERVAL
    )
    return MLCandleReadResult(
        result,
        result[0].open_time if result else None,
        result[-1].open_time if result else None,
        gaps,
    )
