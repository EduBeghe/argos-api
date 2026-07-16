from dataclasses import dataclass
from datetime import UTC, datetime

from app.market.enums import Interval

FIVE_MINUTES_MS = 300_000
FIVE_MINUTES_CLOSE_OFFSET_MS = FIVE_MINUTES_MS - 1
_MILLISECONDS_PER_DAY = 86_400_000
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class BinanceRequestWindow:
    """Inclusive Unix-millisecond boundaries for a Binance kline request."""

    start_time: int
    end_time: int


def latest_closed_five_minute_open_time(now: datetime) -> int:
    """Return the open time of the latest fully closed five-minute candle."""

    now_ms = _utc_datetime_to_unix_milliseconds(now)
    current_open_time = (now_ms // FIVE_MINUTES_MS) * FIVE_MINUTES_MS
    return current_open_time - FIVE_MINUTES_MS


def calculate_closed_candle_window(
    *, now: datetime, interval: Interval, limit: int
) -> BinanceRequestWindow:
    """Calculate a window containing exactly the latest closed candles."""

    if interval is not Interval.FIVE_MINUTES:
        raise ValueError("only the five-minute interval is supported")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")

    latest_open_time = latest_closed_five_minute_open_time(now)
    start_time = latest_open_time - ((limit - 1) * FIVE_MINUTES_MS)
    end_time = latest_open_time + FIVE_MINUTES_CLOSE_OFFSET_MS
    return BinanceRequestWindow(start_time=start_time, end_time=end_time)


def _utc_datetime_to_unix_milliseconds(value: datetime) -> int:
    if value.tzinfo is not UTC:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware UTC")
        raise ValueError("datetime timezone must be UTC")
    if value < _UNIX_EPOCH:
        raise ValueError("datetime must not be before the Unix epoch")

    elapsed = value - _UNIX_EPOCH
    return (
        elapsed.days * _MILLISECONDS_PER_DAY
        + elapsed.seconds * 1000
        + elapsed.microseconds // 1000
    )
