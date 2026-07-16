from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.enums import Asset, Interval
from app.market.models import MarketCandle
from app.market.persistence import MarketCandleRecord, to_domain
from app.market.schemas import CandleOrder


@dataclass(frozen=True, slots=True)
class MarketStatusAggregate:
    """Grouped persistence values for one asset."""

    asset: Asset
    candle_count: int
    latest_open_time: datetime
    latest_close_time: datetime
    latest_ingested_at: datetime


async def list_market_candles(
    *,
    session: AsyncSession,
    asset: Asset,
    interval: Interval,
    start_time: datetime | None,
    end_time: datetime | None,
    limit: int,
    order: CandleOrder,
) -> list[MarketCandle]:
    """Read filtered candles with ordering and limiting performed by PostgreSQL."""

    statement = select(MarketCandleRecord).where(
        MarketCandleRecord.asset == asset.value,
        MarketCandleRecord.interval == interval.value,
    )
    if start_time is not None:
        statement = statement.where(MarketCandleRecord.open_time >= start_time)
    if end_time is not None:
        statement = statement.where(MarketCandleRecord.open_time <= end_time)
    ordering = (
        asc(MarketCandleRecord.open_time)
        if order is CandleOrder.ASC
        else desc(MarketCandleRecord.open_time)
    )
    records = await session.scalars(statement.order_by(ordering).limit(limit))
    return [to_domain(record) for record in records.all()]


async def get_market_status_aggregates(
    *, session: AsyncSession, interval: Interval
) -> list[MarketStatusAggregate]:
    """Read grouped status for all persisted assets in one database query."""

    statement = (
        select(
            MarketCandleRecord.asset,
            func.count(MarketCandleRecord.id),
            func.max(MarketCandleRecord.open_time),
            func.max(MarketCandleRecord.close_time),
            func.max(MarketCandleRecord.ingested_at),
        )
        .where(MarketCandleRecord.interval == interval.value)
        .group_by(MarketCandleRecord.asset)
    )
    rows = (await session.execute(statement)).tuples().all()
    return [
        MarketStatusAggregate(
            asset=Asset(asset),
            candle_count=candle_count,
            latest_open_time=latest_open_time,
            latest_close_time=latest_close_time,
            latest_ingested_at=latest_ingested_at,
        )
        for (
            asset,
            candle_count,
            latest_open_time,
            latest_close_time,
            latest_ingested_at,
        ) in rows
    ]
