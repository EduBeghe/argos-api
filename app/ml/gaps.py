from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from .contract import CANDLE_INTERVAL

NANOSECONDS_PER_SECOND = 1_000_000_000
CANDLE_INTERVAL_NS = int(CANDLE_INTERVAL.total_seconds() * NANOSECONDS_PER_SECOND)


@dataclass(frozen=True, slots=True)
class GapSummary:
    gap_event_count: int
    missing_candle_slot_count: int
    first_missing_open_time: datetime | None
    last_missing_open_time: datetime | None


def datetime_to_ns(value: datetime) -> int:
    return int(value.timestamp() * NANOSECONDS_PER_SECOND)


def ns_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / NANOSECONDS_PER_SECOND, tz=UTC)


def continuity_prefix(open_time_ns: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return cumulative discontinuity events at each candle position."""

    prefix = np.zeros(len(open_time_ns), dtype=np.int64)
    if len(open_time_ns) > 1:
        prefix[1:] = np.cumsum(np.diff(open_time_ns) != CANDLE_INTERVAL_NS)
    return prefix


def summarize_gaps(open_time_ns: NDArray[np.int64]) -> GapSummary:
    if len(open_time_ns) < 2:
        return GapSummary(0, 0, None, None)
    differences = np.diff(open_time_ns)
    gap_positions = np.flatnonzero(differences != CANDLE_INTERVAL_NS)
    if not len(gap_positions):
        return GapSummary(0, 0, None, None)
    missing_by_event = differences[gap_positions] // CANDLE_INTERVAL_NS - 1
    if np.any(missing_by_event <= 0):
        raise ValueError("candle timestamps must be strictly ordered on the 5m grid")
    first_position = int(gap_positions[0])
    last_position = int(gap_positions[-1])
    first_missing = int(open_time_ns[first_position] + CANDLE_INTERVAL_NS)
    last_missing = int(open_time_ns[last_position + 1] - CANDLE_INTERVAL_NS)
    return GapSummary(
        gap_event_count=len(gap_positions),
        missing_candle_slot_count=int(np.sum(missing_by_event)),
        first_missing_open_time=ns_to_datetime(first_missing),
        last_missing_open_time=ns_to_datetime(last_missing),
    )
