from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

from app.paper.models import PaperRunStatus, PaperTradingRun
from app.paper.performance import EquityPoint, calculate_performance

START = datetime(2026, 7, 18, 12, tzinfo=UTC)


def run(initial_cash: Decimal = Decimal("100")) -> PaperTradingRun:
    return PaperTradingRun(
        id=uuid4(),
        status=PaperRunStatus.ACTIVE,
        account_currency="USDT",
        initial_cash=initial_cash,
        current_cash=initial_cash,
        configuration_version=1,
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        started_at=START,
        stopped_at=None,
        stop_reason=None,
        predecessor_run_id=None,
    )


def points(*equities: str) -> tuple[EquityPoint, ...]:
    return tuple(
        EquityPoint(START + timedelta(minutes=30 * index), Decimal(equity))
        for index, equity in enumerate(equities, start=1)
    )


def test_no_snapshots_returns_neutral_baseline() -> None:
    result = calculate_performance(run=run(), snapshots=())
    assert result.snapshot_count == 0
    assert (
        result.latest_equity
        == result.initial_equity
        == Decimal("100.000000000000000000")
    )
    assert result.absolute_pnl == result.total_return == Decimal(0)
    assert result.current_drawdown == result.max_drawdown == Decimal(0)
    assert result.first_snapshot_boundary is None
    assert result.latest_snapshot_boundary is None
    assert result.evaluated_from == result.evaluated_until == START


@pytest.mark.parametrize(
    ("series", "latest", "pnl", "return_value", "current", "maximum"),
    [
        (("110",), "110", "10", "0.1", "0", "0"),
        (("90",), "90", "-10", "-0.1", "0.1", "0.1"),
        (("110", "120", "130"), "130", "30", "0.3", "0", "0"),
        (("90", "80", "70"), "70", "-30", "-0.3", "0.3", "0.3"),
        (("120", "90", "110"), "110", "10", "0.1", "0.083333333333333333", "0.25"),
        (("120", "90", "130"), "130", "30", "0.3", "0", "0.25"),
        (("120", "60", "90"), "90", "-10", "-0.1", "0.25", "0.5"),
    ],
)
def test_performance_series(
    series: tuple[str, ...],
    latest: str,
    pnl: str,
    return_value: str,
    current: str,
    maximum: str,
) -> None:
    result = calculate_performance(run=run(), snapshots=points(*series))
    assert result.latest_equity == Decimal(latest)
    assert result.absolute_pnl == Decimal(pnl)
    assert result.total_return == Decimal(return_value)
    assert result.current_drawdown == Decimal(current)
    assert result.max_drawdown == Decimal(maximum)


def test_snapshots_are_sorted_defensively_and_boundaries_follow_time() -> None:
    ordered = points("120", "80", "110")
    result = calculate_performance(
        run=run(), snapshots=(ordered[2], ordered[0], ordered[1])
    )
    assert result.latest_equity == Decimal("110")
    assert result.first_snapshot_boundary == ordered[0].scheduled_boundary
    assert result.latest_snapshot_boundary == ordered[2].scheduled_boundary
    assert result.evaluated_until == ordered[2].scheduled_boundary


def test_zero_initial_cash_avoids_division_by_zero() -> None:
    result = calculate_performance(run=run(Decimal(0)), snapshots=points("10"))
    assert result.total_return is None
    assert result.peak_equity == result.latest_equity == Decimal("10")
    assert result.lowest_equity == Decimal(0)


def test_values_are_quantized_and_remain_decimal() -> None:
    result = calculate_performance(
        run=run(Decimal("100.0000000000000000004")),
        snapshots=points("90.1234567890123456789"),
    )
    assert result.initial_equity == Decimal("100.000000000000000000")
    assert result.latest_equity == Decimal("90.123456789012345679")
    for value in (
        result.initial_equity,
        result.latest_equity,
        result.absolute_pnl,
        result.total_return,
        result.current_drawdown,
        result.max_drawdown,
    ):
        assert isinstance(value, Decimal)
        exponent = value.as_tuple().exponent
        assert isinstance(exponent, int)
        assert abs(exponent) <= 18


def test_negative_equity_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="negative equity"):
        calculate_performance(run=run(), snapshots=points("-1"))
    with pytest.raises(ValueError, match="initial equity"):
        calculate_performance(
            run=replace(run(), initial_cash=Decimal("-1")), snapshots=()
        )
