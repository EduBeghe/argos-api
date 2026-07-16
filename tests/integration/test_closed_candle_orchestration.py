from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.normalization_errors import BinanceKlineStructureError
from app.ingestion.binance.types import BinanceKlineResponse
from app.ingestion.orchestration import ingest_latest_closed_klines
from app.market.enums import Asset, Interval
from app.market.persistence import MarketCandleRecord

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC)
OPEN_TIME = 1_784_116_800_000


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    database_engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(database_engine, expire_on_commit=False)
    async with factory() as cleanup_session:
        await cleanup_session.execute(delete(MarketCandleRecord))
        await cleanup_session.commit()
    yield database_engine
    async with factory() as cleanup_session:
        await cleanup_session.execute(delete(MarketCandleRecord))
        await cleanup_session.commit()
    await database_engine.dispose()


def raw_kline() -> list[object]:
    """Build the latest closed five-minute kline for NOW."""

    return [
        OPEN_TIME,
        "35000.00",
        "35100.00",
        "34900.00",
        "35050.00",
        "12.50",
        OPEN_TIME + 299_999,
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


async def persisted_assets(engine: AsyncEngine) -> list[str]:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as verification_session:
        assets = await verification_session.scalars(
            select(MarketCandleRecord.asset).order_by(MarketCandleRecord.asset)
        )
        return list(assets)


async def persisted_count(engine: AsyncEngine) -> int:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as verification_session:
        count = await verification_session.scalar(
            select(func.count()).select_from(MarketCandleRecord)
        )
        assert count is not None
        return count


async def test_caller_commits_staged_assets_atomically(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with (
        factory() as session,
        client_for([[raw_kline()], [raw_kline()]]) as client,
    ):
        result = await ingest_latest_closed_klines(
            session=session,
            client=client,
            assets=[Asset.BTC_USDT, Asset.ETH_USDT],
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=1,
        )
        await session.commit()

    assert result.inserted == 2
    assert await persisted_assets(engine) == ["BTC/USDT", "ETH/USDT"]


async def test_caller_rolls_back_staged_assets_on_failure(
    engine: AsyncEngine,
) -> None:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with (
        factory() as session,
        client_for([[raw_kline()], [["invalid"]]]) as client,
    ):
        with pytest.raises(BinanceKlineStructureError):
            await ingest_latest_closed_klines(
                session=session,
                client=client,
                assets=[Asset.BTC_USDT, Asset.ETH_USDT],
                interval=Interval.FIVE_MINUTES,
                now=NOW,
                limit=1,
            )
        await session.rollback()

    assert await persisted_count(engine) == 0


async def test_repeated_orchestration_is_idempotent(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(engine, expire_on_commit=False)
    responses = [[raw_kline()], [raw_kline()], [raw_kline()], [raw_kline()]]
    async with factory() as session, client_for(responses) as client:
        first = await ingest_latest_closed_klines(
            session=session,
            client=client,
            assets=[Asset.BTC_USDT, Asset.ETH_USDT],
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=1,
        )
        await session.commit()
        second = await ingest_latest_closed_klines(
            session=session,
            client=client,
            assets=[Asset.BTC_USDT, Asset.ETH_USDT],
            interval=Interval.FIVE_MINUTES,
            now=NOW,
            limit=1,
        )
        await session.commit()

    assert (first.inserted, first.duplicates) == (2, 0)
    assert (second.inserted, second.duplicates) == (0, 2)
    assert await persisted_count(engine) == 2
