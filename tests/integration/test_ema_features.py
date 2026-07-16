from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.features.calculation import FeatureCalculationResult, calculate_ema_20_features
from app.features.persistence import MarketFeatureRecord
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2026, 7, 15, 10, tzinfo=UTC)
CALCULATED_AT = datetime(2026, 7, 16, tzinfo=UTC)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        await database_session.execute(delete(MarketFeatureRecord))
        await database_session.execute(delete(MarketCandleRecord))
        await database_session.commit()
        yield database_session
        await database_session.rollback()
        await database_session.execute(delete(MarketFeatureRecord))
        await database_session.execute(delete(MarketCandleRecord))
        await database_session.commit()
    await engine.dispose()


async def add_candles(session: AsyncSession, count: int) -> None:
    existing = len((await session.scalars(select(MarketCandleRecord.id))).all())
    for index in range(existing, count):
        open_time = START + timedelta(minutes=5 * index)
        session.add(
            to_record(
                make_candle(
                    open_time=open_time,
                    close_time=open_time + timedelta(minutes=5, milliseconds=-1),
                    close_price=Decimal(index + 1),
                    high_price=Decimal(max(index + 1, 2)),
                    low_price=Decimal(1),
                    open_price=Decimal(1),
                )
            )
        )
    await session.commit()


async def calculate(session: AsyncSession) -> FeatureCalculationResult:
    return await calculate_ema_20_features(
        session=session,
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        calculated_at=CALCULATED_AT,
    )


async def test_ema_feature_history_is_seeded_incremental_and_idempotent(
    session: AsyncSession,
) -> None:
    await add_candles(session, 19)
    insufficient = await calculate(session)
    await session.commit()
    assert insufficient.values_calculated == 0

    await add_candles(session, 20)
    seeded = await calculate(session)
    await session.commit()
    assert (seeded.values_calculated, seeded.values_inserted) == (1, 1)

    repeated = await calculate(session)
    await session.commit()
    assert (repeated.values_inserted, repeated.duplicates) == (0, 1)

    await add_candles(session, 21)
    incremental = await calculate(session)
    await session.commit()
    assert (incremental.values_calculated, incremental.values_inserted) == (2, 1)

    records = (
        await session.scalars(
            select(MarketFeatureRecord).order_by(MarketFeatureRecord.candle_open_time)
        )
    ).all()
    assert len(records) == 2
    assert records[0].value == Decimal("10.5")
    assert records[0].candle_open_time == START + timedelta(minutes=95)
