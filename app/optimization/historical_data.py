import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.market.enums import Interval
from app.market.persistence import MarketCandleRecord

DEFAULT_PAGE_SIZE = 5000
MAX_PAGE_SIZE = 10_000
FEATURE_MODEL_NAME = "baseline_features"
FEATURE_MODEL_VERSION = 1


class HistoricalDatasetIntegrityError(ValueError):
    """Raised when persisted evidence cannot form a strict historical dataset."""


class GapPolicy(StrEnum):
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class HistoricalDatasetRange:
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class HistoricalDatasetIdentity:
    instrument: Instrument
    range: HistoricalDatasetRange
    feature_model_name: str = FEATURE_MODEL_NAME
    feature_model_version: int = FEATURE_MODEL_VERSION


@dataclass(frozen=True, slots=True)
class HistoricalCandle:
    open_time: datetime
    close_time: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    base_asset_volume: Decimal
    quote_asset_volume: Decimal
    trade_count: int
    taker_buy_base_asset_volume: Decimal
    taker_buy_quote_asset_volume: Decimal


@dataclass(frozen=True, slots=True)
class HistoricalDataset:
    identity: HistoricalDatasetIdentity
    expected_candle_count: int
    actual_candle_count: int
    first_open_time: datetime | None
    last_open_time: datetime | None
    gap_count: int
    first_gap: datetime | None
    last_gap: datetime | None
    candles: tuple[HistoricalCandle, ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class HistoricalDatasetRevision:
    """Cheap immutable-candle revision used before loading a full dataset."""

    candle_count: int
    first_open_time: datetime | None
    last_open_time: datetime | None
    latest_ingested_at: datetime | None
    fingerprint: str


async def read_historical_dataset_revision(
    *, session: AsyncSession, identity: HistoricalDatasetIdentity, now: datetime
) -> HistoricalDatasetRevision:
    """Read constant-size range metadata for exact research reuse.

    Market candles are insert-only (`ON CONFLICT DO NOTHING`), so count, temporal
    bounds, and latest ingestion time identify additions to the requested range
    without materializing candle values.
    """

    validate_historical_range(identity, now=now)
    count, first, last, latest_ingested = (
        await session.execute(
            select(
                func.count(MarketCandleRecord.id),
                func.min(MarketCandleRecord.open_time),
                func.max(MarketCandleRecord.open_time),
                func.max(MarketCandleRecord.ingested_at),
            ).where(
                MarketCandleRecord.asset == identity.instrument.asset.value,
                MarketCandleRecord.provider == identity.instrument.provider.value,
                MarketCandleRecord.market_type == identity.instrument.market_type.value,
                MarketCandleRecord.interval == identity.instrument.interval.value,
                MarketCandleRecord.open_time >= identity.range.start,
                MarketCandleRecord.open_time < identity.range.end,
            )
        )
    ).one()
    payload = {
        "instrument": [
            identity.instrument.asset.value,
            identity.instrument.provider.value,
            identity.instrument.market_type.value,
            identity.instrument.interval.value,
        ],
        "range": [identity.range.start.isoformat(), identity.range.end.isoformat()],
        "feature_model": [
            identity.feature_model_name,
            identity.feature_model_version,
        ],
        "candle_count": int(count),
        "first_open_time": first.isoformat() if first else None,
        "last_open_time": last.isoformat() if last else None,
        "latest_ingested_at": latest_ingested.isoformat() if latest_ingested else None,
    }
    fingerprint = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return HistoricalDatasetRevision(
        int(count), first, last, latest_ingested, fingerprint
    )


def interval_duration(interval: Interval) -> timedelta:
    if interval is Interval.FIVE_MINUTES:
        return timedelta(minutes=5)
    raise HistoricalDatasetIntegrityError(
        f"unsupported historical interval: {interval.value}"
    )


def validate_historical_range(
    identity: HistoricalDatasetIdentity, *, now: datetime
) -> None:
    start = identity.range.start
    end = identity.range.end
    if start.tzinfo is not UTC or end.tzinfo is not UTC:
        raise HistoricalDatasetIntegrityError("historical bounds must use UTC")
    if now.tzinfo is not UTC:
        raise HistoricalDatasetIntegrityError("current time must use UTC")
    if start >= end:
        raise HistoricalDatasetIntegrityError("historical range must not be empty")
    step = interval_duration(identity.instrument.interval)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    if (start - epoch) % step or (end - epoch) % step:
        raise HistoricalDatasetIntegrityError(
            "historical bounds must align to the instrument interval"
        )
    current_open = now - ((now - epoch) % step)
    if end > current_open:
        raise HistoricalDatasetIntegrityError(
            "historical range includes an open or future candle"
        )


async def read_historical_dataset(
    *,
    session: AsyncSession,
    identity: HistoricalDatasetIdentity,
    now: datetime,
    page_size: int = DEFAULT_PAGE_SIZE,
    gap_policy: GapPolicy = GapPolicy.REJECT,
) -> HistoricalDataset:
    """Read one strict historical dataset using bounded keyset pages."""

    validate_historical_range(identity, now=now)
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise HistoricalDatasetIntegrityError(
            f"page_size must be between 1 and {MAX_PAGE_SIZE}"
        )
    if gap_policy is not GapPolicy.REJECT:
        raise HistoricalDatasetIntegrityError("unsupported historical gap policy")

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
            MarketCandleRecord.asset == identity.instrument.asset.value,
            MarketCandleRecord.provider == identity.instrument.provider.value,
            MarketCandleRecord.market_type == identity.instrument.market_type.value,
            MarketCandleRecord.interval == identity.instrument.interval.value,
            MarketCandleRecord.open_time >= identity.range.start,
            MarketCandleRecord.open_time < identity.range.end,
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
        _validate_page(page, identity=identity, cursor=cursor)
        candles.extend(page)
        next_cursor = page[-1].open_time
        if cursor is not None and next_cursor <= cursor:
            raise HistoricalDatasetIntegrityError(
                "historical keyset cursor did not advance"
            )
        cursor = next_cursor
        if len(page) < page_size:
            break

    immutable_candles = tuple(candles)
    gaps = _find_gaps(identity, immutable_candles)
    if gaps:
        raise HistoricalDatasetIntegrityError(
            f"historical dataset contains {len(gaps)} missing candle slots; "
            f"first={gaps[0].isoformat()} last={gaps[-1].isoformat()}"
        )
    expected = _expected_count(identity)
    return HistoricalDataset(
        identity=identity,
        expected_candle_count=expected,
        actual_candle_count=len(immutable_candles),
        first_open_time=immutable_candles[0].open_time if immutable_candles else None,
        last_open_time=immutable_candles[-1].open_time if immutable_candles else None,
        gap_count=0,
        first_gap=None,
        last_gap=None,
        candles=immutable_candles,
        fingerprint=dataset_fingerprint(identity, immutable_candles),
    )


def dataset_fingerprint(
    identity: HistoricalDatasetIdentity, candles: tuple[HistoricalCandle, ...]
) -> str:
    canonical_identity = {
        "asset": identity.instrument.asset.value,
        "provider": identity.instrument.provider.value,
        "market_type": identity.instrument.market_type.value,
        "interval": identity.instrument.interval.value,
        "from": identity.range.start.isoformat(),
        "to": identity.range.end.isoformat(),
        "feature_model": identity.feature_model_name,
        "feature_version": identity.feature_model_version,
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(canonical_identity))
    for candle in candles:
        digest.update(
            _canonical_json(
                [
                    candle.open_time.isoformat(),
                    candle.close_time.isoformat(),
                    str(candle.open_price),
                    str(candle.high_price),
                    str(candle.low_price),
                    str(candle.close_price),
                    str(candle.base_asset_volume),
                    str(candle.quote_asset_volume),
                    candle.trade_count,
                    str(candle.taker_buy_base_asset_volume),
                    str(candle.taker_buy_quote_asset_volume),
                ]
            )
        )
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return len(payload).to_bytes(8, "big") + payload


def _validate_page(
    page: tuple[HistoricalCandle, ...],
    *,
    identity: HistoricalDatasetIdentity,
    cursor: datetime | None,
) -> None:
    previous = cursor
    for candle in page:
        if not identity.range.start <= candle.open_time < identity.range.end:
            raise HistoricalDatasetIntegrityError(
                "historical query returned a candle outside the requested range"
            )
        if previous is not None and candle.open_time <= previous:
            reason = "duplicate" if candle.open_time == previous else "out of order"
            raise HistoricalDatasetIntegrityError(
                f"historical dataset contains a {reason} candle timestamp"
            )
        previous = candle.open_time


def _find_gaps(
    identity: HistoricalDatasetIdentity, candles: tuple[HistoricalCandle, ...]
) -> tuple[datetime, ...]:
    available = {candle.open_time for candle in candles}
    step = interval_duration(identity.instrument.interval)
    return tuple(
        identity.range.start + index * step
        for index in range(_expected_count(identity))
        if identity.range.start + index * step not in available
    )


def _expected_count(identity: HistoricalDatasetIdentity) -> int:
    return int(
        (identity.range.end - identity.range.start)
        / interval_duration(identity.instrument.interval)
    )
