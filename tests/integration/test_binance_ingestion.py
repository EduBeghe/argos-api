from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.ingestion import IngestionResult, ingest_klines
from app.ingestion.binance.types import BinanceKlineResponse
from app.market.enums import Asset, Interval
from app.market.persistence import MarketCandleRecord

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

INGESTED_AT = datetime(2026, 7, 15, 18, 30, tzinfo=UTC)


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


def raw_kline(open_time: int) -> list[object]:
    """Build one valid raw Binance five-minute kline."""

    return [
        open_time,
        "35000.00",
        "35100.00",
        "34900.00",
        "35050.00",
        "12.50",
        open_time + 299_999,
        "438125.00",
        100,
        "6.25",
        "219062.50",
        "0",
    ]


def client_for(responses: list[BinanceKlineResponse]) -> BinanceClient:
    queued = iter(responses)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(queued))

    settings = Settings.model_validate(
        {
            "DATABASE_URL": "postgresql+asyncpg://unused:unused@localhost/unused",
            "BINANCE_BASE_URL": "https://binance.test",
        }
    )
    return BinanceClient(settings, transport=httpx.MockTransport(handler))


async def candle_count(session: AsyncSession) -> int:
    count = await session.scalar(select(func.count()).select_from(MarketCandleRecord))
    assert count is not None
    return count


async def run_ingestion(
    session: AsyncSession, client: BinanceClient
) -> IngestionResult:
    return await ingest_klines(
        session=session,
        client=client,
        asset=Asset.BTC_USDT,
        interval=Interval.FIVE_MINUTES,
        ingested_at=INGESTED_AT,
    )


async def test_ingestion_inserts_new_candles(session: AsyncSession) -> None:
    async with client_for(
        [[raw_kline(1_700_000_000_000), raw_kline(1_700_000_300_000)]]
    ) as client:
        result = await run_ingestion(session, client)

    assert result == IngestionResult(2, 2, 2, 0)
    assert await candle_count(session) == 2


async def test_ingestion_ignores_duplicate_candles(session: AsyncSession) -> None:
    row = raw_kline(1_700_000_000_000)
    async with client_for([[row], [row]]) as client:
        first = await run_ingestion(session, client)
        duplicate = await run_ingestion(session, client)

    assert first == IngestionResult(1, 1, 1, 0)
    assert duplicate == IngestionResult(1, 1, 0, 1)
    assert await candle_count(session) == 1


async def test_ingestion_handles_partially_overlapping_candles(
    session: AsyncSession,
) -> None:
    first = raw_kline(1_700_000_000_000)
    overlap = raw_kline(1_700_000_300_000)
    new = raw_kline(1_700_000_600_000)
    async with client_for([[first, overlap], [overlap, new]]) as client:
        initial = await run_ingestion(session, client)
        partial = await run_ingestion(session, client)

    assert initial == IngestionResult(2, 2, 2, 0)
    assert partial == IngestionResult(2, 2, 1, 1)
    assert await candle_count(session) == 3


async def test_ingestion_is_removed_by_caller_rollback(session: AsyncSession) -> None:
    async with client_for([[raw_kline(1_700_000_000_000)]]) as client:
        result = await run_ingestion(session, client)

    assert result.inserted == 1
    assert await candle_count(session) == 1
    await session.rollback()
    assert await candle_count(session) == 0


async def test_empty_response_leaves_database_untouched(session: AsyncSession) -> None:
    row = raw_kline(1_700_000_000_000)
    async with client_for([[row], []]) as client:
        await run_ingestion(session, client)
        before = await candle_count(session)
        empty = await run_ingestion(session, client)
        after = await candle_count(session)

    assert empty == IngestionResult(0, 0, 0, 0)
    assert before == after == 1
