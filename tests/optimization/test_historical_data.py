from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.historical_data import (
    HistoricalCandle,
    HistoricalDatasetIdentity,
    HistoricalDatasetIntegrityError,
    HistoricalDatasetRange,
    dataset_fingerprint,
    read_historical_dataset,
    validate_historical_range,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def identity(count: int = 3) -> HistoricalDatasetIdentity:
    return HistoricalDatasetIdentity(
        INSTRUMENT, HistoricalDatasetRange(START, START + timedelta(minutes=5 * count))
    )


def candle(index: int, *, close: str | None = None) -> HistoricalCandle:
    opened = START + timedelta(minutes=5 * index)
    value = Decimal(close or str(100 + index))
    return HistoricalCandle(
        opened,
        opened + timedelta(minutes=5, milliseconds=-1),
        value,
        value + 1,
        value - 1,
        value,
        Decimal(1),
        Decimal(100),
        10,
        Decimal("0.5"),
        Decimal(50),
    )


def result(rows: list[HistoricalCandle]) -> MagicMock:
    value = MagicMock()
    value.tuples.return_value.all.return_value = [
        tuple(getattr(row, field) for field in row.__dataclass_fields__) for row in rows
    ]
    return value


def test_range_is_utc_aligned_closed_and_half_open() -> None:
    validate_historical_range(identity(), now=START + timedelta(hours=1))
    with pytest.raises(HistoricalDatasetIntegrityError, match="UTC"):
        validate_historical_range(
            HistoricalDatasetIdentity(
                INSTRUMENT,
                HistoricalDatasetRange(
                    START.replace(tzinfo=None), START + timedelta(minutes=5)
                ),
            ),
            now=START + timedelta(hours=1),
        )
    with pytest.raises(HistoricalDatasetIntegrityError, match="align"):
        validate_historical_range(
            HistoricalDatasetIdentity(
                INSTRUMENT, HistoricalDatasetRange(START, START + timedelta(minutes=6))
            ),
            now=START + timedelta(hours=1),
        )
    with pytest.raises(HistoricalDatasetIntegrityError, match="open or future"):
        validate_historical_range(identity(), now=START + timedelta(minutes=14))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("count", "page_size", "calls"), [(4, 5, 1), (5, 5, 2), (6, 5, 2)]
)
async def test_keyset_pages_and_page_size_boundaries(
    count: int, page_size: int, calls: int
) -> None:
    pages = [result([candle(index) for index in range(0, min(count, page_size))])]
    if count >= page_size:
        pages.append(result([candle(index) for index in range(page_size, count)]))
    if count == page_size:
        pages[-1] = result([])
    session = AsyncMock(spec=AsyncSession)
    session.execute.side_effect = pages
    dataset = await read_historical_dataset(
        session=session,
        identity=identity(count),
        now=START + timedelta(days=1),
        page_size=page_size,
    )
    assert dataset.actual_candle_count == dataset.expected_candle_count == count
    assert dataset.first_open_time == START
    assert dataset.last_open_time == START + timedelta(minutes=5 * (count - 1))
    assert session.execute.await_count == calls
    statements = [str(item.args[0]) for item in session.execute.await_args_list]
    assert all(
        "LIMIT" in statement and "OFFSET" not in statement for statement in statements
    )
    if calls > 1:
        assert "market_candles.open_time >" in statements[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "message"),
    [([candle(0), candle(0)], "duplicate"), ([candle(1), candle(0)], "out of order")],
)
async def test_rejects_duplicate_or_disordered_rows(
    rows: list[HistoricalCandle], message: str
) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.execute.return_value = result(rows)
    with pytest.raises(HistoricalDatasetIntegrityError, match=message):
        await read_historical_dataset(
            session=session, identity=identity(2), now=START + timedelta(days=1)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows", [[candle(1), candle(2)], [candle(0), candle(2)], [candle(0), candle(1)]]
)
async def test_rejects_initial_internal_and_final_gaps(
    rows: list[HistoricalCandle],
) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.execute.return_value = result(rows)
    with pytest.raises(HistoricalDatasetIntegrityError, match="missing candle slots"):
        await read_historical_dataset(
            session=session, identity=identity(3), now=START + timedelta(days=1)
        )


def test_fingerprint_is_deterministic_and_sensitive_to_candle_values() -> None:
    original = tuple(candle(index) for index in range(3))
    assert dataset_fingerprint(identity(), original) == dataset_fingerprint(
        identity(), original
    )
    changed = (original[0], candle(1, close="999"), original[2])
    assert dataset_fingerprint(identity(), original) != dataset_fingerprint(
        identity(), changed
    )
