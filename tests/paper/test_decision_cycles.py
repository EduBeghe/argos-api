from datetime import UTC, datetime

import pytest

from app.paper.decision_cycles import (
    DecisionCycleStartDelayError,
    latest_completed_boundary,
    validate_cycle_start,
)


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        ((2026, 7, 17, 12, 0), (2026, 7, 17, 12, 0)),
        ((2026, 7, 17, 12, 2), (2026, 7, 17, 12, 0)),
        ((2026, 7, 17, 12, 29), (2026, 7, 17, 12, 0)),
        ((2026, 7, 17, 12, 30), (2026, 7, 17, 12, 30)),
        ((2026, 7, 17, 12, 32), (2026, 7, 17, 12, 30)),
        ((2026, 7, 18, 0, 1), (2026, 7, 18, 0, 0)),
    ],
)
def test_latest_completed_boundary(
    moment: tuple[int, int, int, int, int],
    expected: tuple[int, int, int, int, int],
) -> None:
    assert latest_completed_boundary(datetime(*moment, tzinfo=UTC)) == datetime(
        *expected, tzinfo=UTC
    )


def test_boundary_requires_aware_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        latest_completed_boundary(datetime(2026, 7, 17, 12))


@pytest.mark.parametrize(
    ("boundary", "execution_time"),
    [
        ((2026, 7, 17, 12, 0), (2026, 7, 17, 12, 2)),
        ((2026, 7, 17, 12, 0), (2026, 7, 17, 12, 10)),
        ((2026, 7, 17, 12, 30), (2026, 7, 17, 12, 32)),
        ((2026, 7, 18, 0, 0), (2026, 7, 18, 0, 2)),
    ],
)
def test_cycle_start_accepts_execution_window(
    boundary: tuple[int, int, int, int, int],
    execution_time: tuple[int, int, int, int, int],
) -> None:
    validate_cycle_start(
        boundary=datetime(*boundary, tzinfo=UTC),
        execution_time=datetime(*execution_time, tzinfo=UTC),
    )


@pytest.mark.parametrize("minute", [11, 29])
def test_cycle_start_rejects_excessive_delay(minute: int) -> None:
    with pytest.raises(DecisionCycleStartDelayError, match="maximum delay"):
        validate_cycle_start(
            boundary=datetime(2026, 7, 17, 12, tzinfo=UTC),
            execution_time=datetime(2026, 7, 17, 12, minute, tzinfo=UTC),
        )


def test_cycle_start_rejects_execution_before_boundary() -> None:
    with pytest.raises(DecisionCycleStartDelayError, match="must not precede"):
        validate_cycle_start(
            boundary=datetime(2026, 7, 17, 12, 30, tzinfo=UTC),
            execution_time=datetime(2026, 7, 17, 12, 29, tzinfo=UTC),
        )

