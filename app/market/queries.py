from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import asc, desc, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.models import MarketCandle
from app.market.persistence import MarketCandleRecord, to_domain
from app.market.schemas import CandleOrder


@dataclass(frozen=True, slots=True)
class MarketStatusAggregate:
    """Grouped persistence values for one complete instrument identity."""

    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    candle_count: int
    latest_open_time: datetime
    latest_close_time: datetime
    latest_ingested_at: datetime


async def list_market_candles(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    start_time: datetime | None,
    end_time: datetime | None,
    limit: int,
    order: CandleOrder,
) -> list[MarketCandle]:
    """Read filtered candles with ordering and limiting performed by PostgreSQL."""

    statement = select(MarketCandleRecord).where(
        MarketCandleRecord.asset == asset.value,
        MarketCandleRecord.provider == provider.value,
        MarketCandleRecord.market_type == market_type.value,
        MarketCandleRecord.interval == interval.value,
    )
    if start_time is not None:
        statement = statement.where(MarketCandleRecord.open_time >= start_time)
    if end_time is not None:
        statement = statement.where(MarketCandleRecord.open_time <= end_time)
    ordering = (
        (asc(MarketCandleRecord.open_time), asc(MarketCandleRecord.id))
        if order is CandleOrder.ASC
        else (desc(MarketCandleRecord.open_time), desc(MarketCandleRecord.id))
    )
    records = await session.scalars(statement.order_by(*ordering).limit(limit))
    return [to_domain(record) for record in records.all()]


async def get_market_status_aggregates(
    *,
    session: AsyncSession,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
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
        .where(
            MarketCandleRecord.provider == provider.value,
            MarketCandleRecord.market_type == market_type.value,
            MarketCandleRecord.interval == interval.value,
        )
        .group_by(MarketCandleRecord.asset)
    )
    rows = (await session.execute(statement)).tuples().all()
    return [
        MarketStatusAggregate(
            asset=Asset(asset),
            provider=provider,
            market_type=market_type,
            interval=interval,
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


async def get_market_status_aggregates_batch(
    *, session: AsyncSession, instruments: Sequence[Instrument]
) -> tuple[MarketStatusAggregate, ...]:
    """Read status only for requested complete identities in one query."""

    if not instruments:
        return ()
    identities = [
        (
            item.asset.value,
            item.provider.value,
            item.market_type.value,
            item.interval.value,
        )
        for item in instruments
    ]
    statement = (
        select(
            MarketCandleRecord.asset,
            MarketCandleRecord.provider,
            MarketCandleRecord.market_type,
            MarketCandleRecord.interval,
            func.count(MarketCandleRecord.id),
            func.max(MarketCandleRecord.open_time),
            func.max(MarketCandleRecord.close_time),
            func.max(MarketCandleRecord.ingested_at),
        )
        .where(
            tuple_(
                MarketCandleRecord.asset,
                MarketCandleRecord.provider,
                MarketCandleRecord.market_type,
                MarketCandleRecord.interval,
            ).in_(identities)
        )
        .group_by(
            MarketCandleRecord.asset,
            MarketCandleRecord.provider,
            MarketCandleRecord.market_type,
            MarketCandleRecord.interval,
        )
    )
    rows = (await session.execute(statement)).tuples().all()
    return tuple(
        MarketStatusAggregate(
            asset=Asset(asset),
            provider=Provider(provider),
            market_type=MarketType(market_type),
            interval=Interval(interval),
            candle_count=candle_count,
            latest_open_time=latest_open_time,
            latest_close_time=latest_close_time,
            latest_ingested_at=latest_ingested_at,
        )
        for (
            asset,
            provider,
            market_type,
            interval,
            candle_count,
            latest_open_time,
            latest_close_time,
            latest_ingested_at,
        ) in rows
    )
