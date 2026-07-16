from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.market.api import build_market_status
from app.market.enums import Asset, Interval
from app.market.persistence import MarketCandleRecord, to_record
from app.market.queries import get_market_status_aggregates, list_market_candles
from app.market.schemas import CandleOrder
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

BASE_OPEN = datetime(2026, 7, 15, 11, 50, tzinfo=UTC)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        await database_session.execute(delete(MarketCandleRecord))
        await database_session.commit()
        yield database_session
        await database_session.rollback()
        await database_session.execute(delete(MarketCandleRecord))
        await database_session.commit()
    await engine.dispose()


async def insert_btc_candles(session: AsyncSession) -> None:
    for offset in (0, 5, 10):
        open_time = BASE_OPEN + timedelta(minutes=offset)
        session.add(
            to_record(
                make_candle(
                    open_time=open_time,
                    close_time=open_time + timedelta(minutes=5, milliseconds=-1),
                    ingested_at=open_time + timedelta(minutes=5, seconds=1),
                )
            )
        )
    await session.commit()


async def test_candle_query_orders_ranges_and_limits_in_postgresql(
    session: AsyncSession,
) -> None:
    await insert_btc_candles(session)

    latest = await list_market_candles(
        session=session,
        asset=Asset.BTC_USDT,
        interval=Interval.FIVE_MINUTES,
        start_time=None,
        end_time=None,
        limit=2,
        order=CandleOrder.DESC,
    )
    ranged = await list_market_candles(
        session=session,
        asset=Asset.BTC_USDT,
        interval=Interval.FIVE_MINUTES,
        start_time=BASE_OPEN + timedelta(minutes=5),
        end_time=BASE_OPEN + timedelta(minutes=10),
        limit=100,
        order=CandleOrder.ASC,
    )

    assert [candle.open_time for candle in latest] == [
        BASE_OPEN + timedelta(minutes=10),
        BASE_OPEN + timedelta(minutes=5),
    ]
    assert [candle.open_time for candle in ranged] == [
        BASE_OPEN + timedelta(minutes=5),
        BASE_OPEN + timedelta(minutes=10),
    ]


async def test_grouped_status_supports_missing_asset_and_freshness(
    session: AsyncSession,
) -> None:
    await insert_btc_candles(session)

    aggregates = await get_market_status_aggregates(
        session=session, interval=Interval.FIVE_MINUTES
    )
    statuses = build_market_status(
        aggregates=aggregates,
        now=datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC),
    )

    assert len(aggregates) == 1
    assert aggregates[0].candle_count == 3
    assert aggregates[0].latest_open_time == datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    assert [status.asset for status in statuses] == [
        Asset.BTC_USDT,
        Asset.ETH_USDT,
    ]
    assert statuses[0].is_fresh is True
    assert statuses[1].candle_count == 0
    assert statuses[1].is_fresh is False
