from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.ml.gaps import datetime_to_ns, summarize_gaps

START = datetime(2023, 1, 1, tzinfo=UTC)


def timestamps(indices: list[int]) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    return np.asarray(
        [datetime_to_ns(START + timedelta(minutes=5 * index)) for index in indices],
        dtype=np.int64,
    )


@pytest.mark.parametrize(
    ("indices", "events", "slots"),
    [
        (list(range(5)), 0, 0),
        ([0, 2], 1, 1),
        ([0, 17], 1, 16),
        ([0, 2, 3, 6], 2, 3),
    ],
)
def test_gap_events_and_missing_slots_are_distinct(
    indices: list[int], events: int, slots: int
) -> None:
    summary = summarize_gaps(timestamps(indices))
    assert summary.gap_event_count == events
    assert summary.missing_candle_slot_count == slots


def test_known_sixteen_slot_outage_has_exact_bounds() -> None:
    start = datetime(2023, 1, 1, 12, 35, tzinfo=UTC)
    values = np.asarray(
        [datetime_to_ns(start), datetime_to_ns(start + timedelta(minutes=85))],
        dtype=np.int64,
    )
    summary = summarize_gaps(values)
    assert summary.first_missing_open_time == datetime(2023, 1, 1, 12, 40, tzinfo=UTC)
    assert summary.last_missing_open_time == datetime(2023, 1, 1, 13, 55, tzinfo=UTC)
