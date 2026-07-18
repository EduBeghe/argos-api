from datetime import UTC, datetime

import pytest

from app.paper.decision_cycles import latest_completed_boundary


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
