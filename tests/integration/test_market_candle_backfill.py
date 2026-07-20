Exit code: 0
Wall time: 0.4 seconds
Output:
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.ingestion.backfill import BackfillRange, backfill_instrument
from app.ingestion.binance.client import BinanceClient
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord
from app.market.storage_telemetry import read_market_storage_telemetry

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2024, 2, 1, tzinfo=UTC)
END = START + timedelta(minutes=15)
INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def row(open_time: datetime) -> list[object]:
    milliseconds = int(open_time.timestamp() * 1000)
    return [
        milliseconds,
        "100.123",
        "101",
        "99",
        "100",
        "1",
        milliseconds + 299_999,
        "100",
        1,
        "0.5",
        "50",
        "0",
    ]


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as value:
        cleanup = delete(MarketCandleRecord).where(
            MarketCandleRecord.open_time >= START, MarketCandleRecord.open_time < END
        )
        await value.execute(cleanup)
        await value.commit()
        yield value
        await value.rollback()
        await value.execute(cleanup)
        await value.commit()
    await engine.dispose()


def client(responses: list[list[list[object]]]) -> BinanceClient:
    queued = iter(responses)

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(queued))

    settings = Settings.model_validate(
        {
            "DATABASE_URL": get_settings().database_url,
            "BINANCE_BASE_URL": "https://binance.test",
        }
    )
    return BinanceClient(settings, transport=httpx.MockTransport(handler))


async def count_rows(session: AsyncSession) -> int:
    value = await session.scalar(
        select(func.count())
        .select_from(MarketCandleRecord)
        .where(
            MarketCandleRecord.open_time >= START, MarketCandleRecord.open_time < END
        )
    )
    return int(value or 0)


async def test_backfill_is_idempotent_and_coexists_with_existing_candles(
    session: AsyncSession,
) -> None:
    first = row(START)
    rest = [row(START + timedelta(minutes=5)), row(START + timedelta(minutes=10))]
    async with client([[first]]) as normal_client:
        await backfill_instrument(
            session=session,
            client=normal_client,
            instrument=INSTRUMENT,
            requested=BackfillRange(
                START, START + timedelta(minutes=5), Interval.FIVE_MINUTES
            ),
            ingested_at=END,
        )

    async with client([[first, *rest], [first, *rest]]) as backfill_client:
        initial = await backfill_instrument(
            session=session,
            client=backfill_client,
            instrument=INSTRUMENT,
            requested=BackfillRange(START, END, Interval.FIVE_MINUTES),
            ingested_at=END,
        )
        repeated = await backfill_instrument(
            session=session,
            client=backfill_client,
            instrument=INSTRUMENT,
            requested=BackfillRange(START, END, Interval.FIVE_MINUTES),
            ingested_at=END,
        )

    assert (initial.inserted, initial.duplicates) == (2, 1)
    assert (repeated.inserted, repeated.duplicates) == (0, 3)
    assert repeated.missing == 0
    assert await count_rows(session) == 3
    telemetry = await read_market_storage_telemetry(session)
    assert telemetry.supported is True
    assert telemetry.rows is not None and telemetry.rows >= 3


async def test_completed_pages_survive_interruption_and_resume(
    session: AsyncSession,
) -> None:
    calls = 0

    async def interrupted(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise httpx.ConnectError("interrupted")
        return httpx.Response(200, json=[row(START)])

    settings = Settings.model_validate(
        {
            "DATABASE_URL": get_settings().database_url,
            "BINANCE_BASE_URL": "https://binance.test",
        }
    )
    async with BinanceClient(
        settings, transport=httpx.MockTransport(interrupted)
    ) as failing:
        with pytest.raises(Exception, match="failed"):
            await backfill_instrument(
                session=session,
                client=failing,
                instrument=INSTRUMENT,
                requested=BackfillRange(START, END, Interval.FIVE_MINUTES),
                ingested_at=END,
                page_size=1,
                max_retries=0,
            )
    assert await count_rows(session) == 1

    remaining = [
        row(START),
        row(START + timedelta(minutes=5)),
        row(START + timedelta(minutes=10)),
    ]
    async with client([[remaining[0]], [remaining[1]], [remaining[2]]]) as retry_client:
        result = await backfill_instrument(
            session=session,
            client=retry_client,
            instrument=INSTRUMENT,
            requested=BackfillRange(START, END, Interval.FIVE_MINUTES),
            ingested_at=END,
            page_size=1,
        )
    assert (result.inserted, result.duplicates, result.missing) == (2, 1, 0)
    assert await count_rows(session) == 3

