from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.ingestion.backfill import (
    BackfillError,
    BackfillRange,
    backfill_instrument,
    validate_closed_range,
)
from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.errors import BinanceHTTPError
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider

START = datetime(2024, 1, 1, tzinfo=UTC)
INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def raw_kline(open_time: datetime) -> list[object]:
    milliseconds = int(open_time.timestamp() * 1000)
    return [
        milliseconds,
        "100",
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


def client_with(transport: httpx.AsyncBaseTransport) -> BinanceClient:
    settings = Settings.model_validate(
        {
            "DATABASE_URL": "postgresql+asyncpg://unused/unused",
            "BINANCE_BASE_URL": "https://binance.test",
        }
    )
    return BinanceClient(settings, transport=transport)


def test_range_requires_utc_order_and_alignment() -> None:
    assert (
        BackfillRange(
            START, START + timedelta(minutes=5), Interval.FIVE_MINUTES
        ).expected_slots
        == 1
    )
    with pytest.raises(BackfillError, match="UTC"):
        BackfillRange(
            START.replace(tzinfo=None),
            START + timedelta(minutes=5),
            Interval.FIVE_MINUTES,
        )
    with pytest.raises(BackfillError, match="earlier"):
        BackfillRange(START, START, Interval.FIVE_MINUTES)
    with pytest.raises(BackfillError, match="aligned"):
        BackfillRange(START, START + timedelta(minutes=6), Interval.FIVE_MINUTES)


def test_range_rejects_open_candles() -> None:
    requested = BackfillRange(
        START, START + timedelta(minutes=10), Interval.FIVE_MINUTES
    )
    validate_closed_range(requested, now=START + timedelta(minutes=10))
    with pytest.raises(BackfillError, match="open or future"):
        validate_closed_range(requested, now=START + timedelta(minutes=9))


@pytest.mark.parametrize(("slots", "pages"), [(999, 1), (1000, 1), (1001, 2)])
@pytest.mark.asyncio
async def test_dry_run_pages_inclusive_exclusive(slots: int, pages: int) -> None:
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        start_ms = int(request.url.params["startTime"])
        limit = int(request.url.params["limit"])
        end_ms = int(request.url.params["endTime"])
        rows = [
            raw_kline(datetime.fromtimestamp(value / 1000, tz=UTC))
            for value in range(
                start_ms, min(end_ms + 1, start_ms + limit * 300_000), 300_000
            )
        ]
        return httpx.Response(200, json=rows)

    requested = BackfillRange(
        START, START + slots * timedelta(minutes=5), Interval.FIVE_MINUTES
    )
    session = AsyncMock(spec=AsyncSession)
    async with client_with(httpx.MockTransport(handler)) as client:
        result = await backfill_instrument(
            session=session,
            client=client,
            instrument=INSTRUMENT,
            requested=requested,
            ingested_at=START,
            dry_run=True,
        )
    assert len(calls) == pages
    assert result.expected_slots == result.received == slots
    assert result.missing == 0
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_max_pages_and_invalid_pages() -> None:
    async def one(request: httpx.Request) -> httpx.Response:
        start = datetime.fromtimestamp(
            int(request.url.params["startTime"]) / 1000, tz=UTC
        )
        return httpx.Response(200, json=[raw_kline(start)])

    async with client_with(httpx.MockTransport(one)) as client:
        result = await backfill_instrument(
            session=AsyncMock(spec=AsyncSession),
            client=client,
            instrument=INSTRUMENT,
            requested=BackfillRange(
                START, START + timedelta(minutes=15), Interval.FIVE_MINUTES
            ),
            ingested_at=START,
            page_size=1,
            max_pages=2,
            dry_run=True,
        )
    assert (result.pages, result.received, result.missing) == (2, 2, 1)

    outside = raw_kline(START - timedelta(minutes=5))
    async with client_with(
        httpx.MockTransport(lambda _: httpx.Response(200, json=[outside]))
    ) as client:
        with pytest.raises(BackfillError, match="outside"):
            await backfill_instrument(
                session=AsyncMock(spec=AsyncSession),
                client=client,
                instrument=INSTRUMENT,
                requested=BackfillRange(
                    START, START + timedelta(minutes=5), Interval.FIVE_MINUTES
                ),
                ingested_at=START,
                dry_run=True,
            )

    repeated = [raw_kline(START), raw_kline(START)]
    async with client_with(
        httpx.MockTransport(lambda _: httpx.Response(200, json=repeated))
    ) as client:
        with pytest.raises(BackfillError, match="strictly increasing"):
            await backfill_instrument(
                session=AsyncMock(spec=AsyncSession),
                client=client,
                instrument=INSTRUMENT,
                requested=BackfillRange(
                    START, START + timedelta(minutes=10), Interval.FIVE_MINUTES
                ),
                ingested_at=START,
                dry_run=True,
            )


@pytest.mark.asyncio
async def test_retries_transient_errors_but_not_permanent_4xx() -> None:
    attempts = 0
    sleeps: list[float] = []

    async def transient(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return (
            httpx.Response(503, json={"msg": "busy"})
            if attempts < 3
            else httpx.Response(200, json=[raw_kline(START)])
        )

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    async with client_with(httpx.MockTransport(transient)) as client:
        result = await backfill_instrument(
            session=AsyncMock(spec=AsyncSession),
            client=client,
            instrument=INSTRUMENT,
            requested=BackfillRange(
                START, START + timedelta(minutes=5), Interval.FIVE_MINUTES
            ),
            ingested_at=START,
            dry_run=True,
            sleep=sleep,
        )
    assert result.retries == 2
    assert sleeps == [1.0, 2.0]

    async with client_with(
        httpx.MockTransport(lambda _: httpx.Response(400, json={"msg": "bad"}))
    ) as client:
        with pytest.raises(BinanceHTTPError):
            await backfill_instrument(
                session=AsyncMock(spec=AsyncSession),
                client=client,
                instrument=INSTRUMENT,
                requested=BackfillRange(
                    START, START + timedelta(minutes=5), Interval.FIVE_MINUTES
                ),
                ingested_at=START,
                dry_run=True,
                sleep=sleep,
            )
