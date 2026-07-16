from datetime import UTC, datetime, timedelta, timezone
from typing import cast

import pytest

from app.ingestion.time_windows import (
    FIVE_MINUTES_MS,
    calculate_closed_candle_window,
    latest_closed_five_minute_open_time,
)
from app.market.enums import Interval


def utc_milliseconds(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = value - epoch
    return (
        elapsed.days * 86_400_000
        + elapsed.seconds * 1000
        + elapsed.microseconds // 1000
    )


@pytest.mark.parametrize(
    ("now", "expected_open"),
    [
        (
            datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC),
            datetime(2026, 7, 15, 12, 0, tzinfo=UTC),
        ),
        (
            datetime(2026, 7, 15, 12, 5, tzinfo=UTC),
            datetime(2026, 7, 15, 12, 0, tzinfo=UTC),
        ),
        (
            datetime(2026, 7, 15, 12, 4, 59, 999000, tzinfo=UTC),
            datetime(2026, 7, 15, 11, 55, tzinfo=UTC),
        ),
        (
            datetime(2026, 7, 15, 12, 10, 0, 500, tzinfo=UTC),
            datetime(2026, 7, 15, 12, 5, tzinfo=UTC),
        ),
    ],
)
def test_latest_closed_candle_respects_five_minute_boundaries(
    now: datetime, expected_open: datetime
) -> None:
    assert latest_closed_five_minute_open_time(now) == utc_milliseconds(expected_open)


@pytest.mark.parametrize("limit", [1, 2, 17, 1000])
def test_window_contains_exactly_requested_closed_candles(limit: int) -> None:
    now = datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC)

    window = calculate_closed_candle_window(
        now=now,
        interval=Interval.FIVE_MINUTES,
        limit=limit,
    )

    latest_open = utc_milliseconds(datetime(2026, 7, 15, 12, 0, tzinfo=UTC))
    assert window.start_time == latest_open - ((limit - 1) * FIVE_MINUTES_MS)
    assert window.end_time == latest_open + FIVE_MINUTES_MS - 1
    assert (window.end_time - window.start_time + 1) // FIVE_MINUTES_MS == limit


def test_window_for_two_candles_matches_documented_example() -> None:
    window = calculate_closed_candle_window(
        now=datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC),
        interval=Interval.FIVE_MINUTES,
        limit=2,
    )

    assert window.start_time == utc_milliseconds(
        datetime(2026, 7, 15, 11, 55, tzinfo=UTC)
    )
    assert window.end_time == utc_milliseconds(
        datetime(2026, 7, 15, 12, 4, 59, 999000, tzinfo=UTC)
    )


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 7, 15, 12, 7, 30),
        datetime(2026, 7, 15, 9, 7, 30, tzinfo=timezone(timedelta(hours=-3))),
    ],
)
def test_window_rejects_non_utc_datetime(now: datetime) -> None:
    with pytest.raises(ValueError, match="UTC"):
        calculate_closed_candle_window(
            now=now,
            interval=Interval.FIVE_MINUTES,
            limit=1,
        )


def test_window_rejects_unsupported_interval() -> None:
    with pytest.raises(ValueError, match="five-minute"):
        calculate_closed_candle_window(
            now=datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC),
            interval=cast(Interval, "1m"),
            limit=1,
        )


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_window_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit"):
        calculate_closed_candle_window(
            now=datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC),
            interval=Interval.FIVE_MINUTES,
            limit=limit,
        )
