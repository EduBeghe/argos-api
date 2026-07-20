import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.binance.client import BinanceClient
from app.ingestion.binance.errors import (
    BinanceHTTPError,
    BinanceTimeoutError,
    BinanceTransportError,
)
from app.ingestion.binance.normalizer import normalize_kline
from app.ingestion.binance.types import BinanceKlineResponse
from app.market.catalog import Instrument
from app.market.enums import Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, insert_market_candles

MAX_PAGE_SIZE = 1000
DEFAULT_PAGE_SIZE = 1000
DEFAULT_MAX_RETRIES = 3
FIVE_MINUTES = timedelta(minutes=5)
Sleep = Callable[[float], Awaitable[None]]


class BackfillError(ValueError):
    """Raised when a backfill request or provider page is unsafe."""


@dataclass(frozen=True, slots=True)
class BackfillRange:
    start: datetime
    end: datetime
    interval: Interval

    def __post_init__(self) -> None:
        for name, value in (("from", self.start), ("to", self.end)):
            if value.tzinfo is not UTC:
                raise BackfillError(f"{name} must use the UTC timezone")
        if self.start >= self.end:
            raise BackfillError("from must be earlier than to")
        step = interval_duration(self.interval)
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        if (self.start - epoch) % step or (self.end - epoch) % step:
            raise BackfillError("from and to must be aligned to the candle interval")

    @property
    def expected_slots(self) -> int:
        return int((self.end - self.start) / interval_duration(self.interval))


@dataclass(frozen=True, slots=True)
class BackfillResult:
    instrument: Instrument
    expected_slots: int
    received: int
    inserted: int
    duplicates: int
    missing: int
    first_missing_open_time: datetime | None
    last_missing_open_time: datetime | None
    pages: int
    retries: int
    first_processed_open_time: datetime | None
    last_processed_open_time: datetime | None


def interval_duration(interval: Interval) -> timedelta:
    if interval is Interval.FIVE_MINUTES:
        return FIVE_MINUTES
    raise BackfillError(f"unsupported backfill interval: {interval.value}")


def validate_closed_range(requested: BackfillRange, *, now: datetime) -> None:
    if now.tzinfo is not UTC:
        raise BackfillError("current time must use the UTC timezone")
    step = interval_duration(requested.interval)
    current_open = now - ((now - datetime(1970, 1, 1, tzinfo=UTC)) % step)
    if requested.end > current_open:
        raise BackfillError("range includes an open or future candle")


async def backfill_instrument(
    *,
    session: AsyncSession,
    client: BinanceClient,
    instrument: Instrument,
    requested: BackfillRange,
    ingested_at: datetime,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_pages: int | None = None,
    dry_run: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep: Sleep = asyncio.sleep,
    page_completed: Callable[[int, int, int, int], None] | None = None,
) -> BackfillResult:
    """Import one catalog instrument from oldest to newest, committing each page."""

    _validate_options(instrument, requested, page_size, max_pages, max_retries)
    step = interval_duration(requested.interval)
    cursor = requested.start
    received = inserted = duplicates = pages = retries = 0
    processed: set[datetime] = set()

    while cursor < requested.end and (max_pages is None or pages < max_pages):
        remaining = int((requested.end - cursor) / step)
        requested_limit = min(page_size, remaining)
        raw_page, page_retries = await _get_page_with_retries(
            client=client,
            instrument=instrument,
            start_time=_to_milliseconds(cursor),
            end_time=_to_milliseconds(requested.end) - 1,
            limit=requested_limit,
            max_retries=max_retries,
            sleep=sleep,
        )
        retries += page_retries
        if not raw_page:
            break
        open_times = _validate_page(raw_page, cursor=cursor, requested=requested)
        candles = [
            normalize_kline(
                raw_kline=row,
                asset=instrument.asset,
                interval=instrument.interval,
                ingested_at=ingested_at,
            )
            for row in raw_page
        ]
        page_inserted = 0
        if not dry_run:
            try:
                page_inserted = await insert_market_candles(session, candles)
                await session.commit()
            except Exception:
                await session.rollback()
                raise
        page_duplicates = len(candles) - page_inserted if not dry_run else 0
        received += len(candles)
        inserted += page_inserted
        duplicates += page_duplicates
        pages += 1
        processed.update(open_times)
        if page_completed is not None:
            page_completed(pages, len(candles), page_inserted, page_duplicates)
        next_cursor = open_times[-1] + step
        if next_cursor <= cursor:
            raise BackfillError("Binance pagination cursor did not advance")
        cursor = next_cursor

    available = (
        processed
        if dry_run
        else await _persisted_open_times(
            session, instrument=instrument, requested=requested
        )
    )
    expected = tuple(_expected_open_times(requested))
    missing_times = tuple(value for value in expected if value not in available)
    ordered_processed = sorted(processed)
    return BackfillResult(
        instrument=instrument,
        expected_slots=len(expected),
        received=received,
        inserted=inserted,
        duplicates=duplicates,
        missing=len(missing_times),
        first_missing_open_time=missing_times[0] if missing_times else None,
        last_missing_open_time=missing_times[-1] if missing_times else None,
        pages=pages,
        retries=retries,
        first_processed_open_time=ordered_processed[0] if ordered_processed else None,
        last_processed_open_time=ordered_processed[-1] if ordered_processed else None,
    )


def _validate_options(
    instrument: Instrument,
    requested: BackfillRange,
    page_size: int,
    max_pages: int | None,
    max_retries: int,
) -> None:
    if (
        instrument.provider is not Provider.BINANCE
        or instrument.market_type is not MarketType.SPOT
    ):
        raise BackfillError("the current backfill supports Binance Spot only")
    if instrument.interval is not requested.interval:
        raise BackfillError("instrument interval does not match the requested range")
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise BackfillError("page_size must be between 1 and 1000")
    if max_pages is not None and max_pages < 1:
        raise BackfillError("max_pages must be positive")
    if max_retries < 0:
        raise BackfillError("max_retries must be non-negative")


async def _get_page_with_retries(
    *,
    client: BinanceClient,
    instrument: Instrument,
    start_time: int,
    end_time: int,
    limit: int,
    max_retries: int,
    sleep: Sleep,
) -> tuple[BinanceKlineResponse, int]:
    retries = 0
    while True:
        try:
            return await client.get_klines(
                asset=instrument.asset,
                interval=instrument.interval,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            ), retries
        except (BinanceTimeoutError, BinanceTransportError):
            if retries >= max_retries:
                raise
        except BinanceHTTPError as error:
            if error.status_code != 429 and not 500 <= error.status_code < 600:
                raise
            if retries >= max_retries:
                raise
        retries += 1
        await sleep(float(2 ** (retries - 1)))


def _validate_page(
    page: BinanceKlineResponse, *, cursor: datetime, requested: BackfillRange
) -> tuple[datetime, ...]:
    open_times: list[datetime] = []
    for row in page:
        if not row or not isinstance(row[0], int) or isinstance(row[0], bool):
            raise BackfillError("Binance page contains an invalid open time")
        open_time = datetime.fromtimestamp(row[0] / 1000, tz=UTC)
        if (
            open_time < cursor
            or open_time < requested.start
            or open_time >= requested.end
        ):
            raise BackfillError(
                "Binance page contains a candle outside the requested range"
            )
        if open_times and open_time <= open_times[-1]:
            raise BackfillError("Binance page open times are not strictly increasing")
        open_times.append(open_time)
    return tuple(open_times)


async def _persisted_open_times(
    session: AsyncSession, *, instrument: Instrument, requested: BackfillRange
) -> set[datetime]:
    statement = select(MarketCandleRecord.open_time).where(
        MarketCandleRecord.asset == instrument.asset.value,
        MarketCandleRecord.provider == instrument.provider.value,
        MarketCandleRecord.market_type == instrument.market_type.value,
        MarketCandleRecord.interval == instrument.interval.value,
        MarketCandleRecord.open_time >= requested.start,
        MarketCandleRecord.open_time < requested.end,
    )
    return set((await session.scalars(statement)).all())


def _expected_open_times(requested: BackfillRange) -> Sequence[datetime]:
    step = interval_duration(requested.interval)
    count = requested.expected_slots
    return tuple(requested.start + index * step for index in range(count))


def _to_milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)
