from decimal import Decimal

import pytest

from app.features.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_relative_volume,
    calculate_return,
    calculate_rsi,
)


def decimals(values: range) -> list[Decimal]:
    return [Decimal(value) for value in values]


def test_ema_50_alignment_seed_recursion_and_no_future_influence() -> None:
    assert calculate_ema(decimals(range(1, 50)), period=50) == []
    points = calculate_ema(decimals(range(1, 52)), period=50)
    assert points[0].candle_index == 49
    assert points[0].value == Decimal("25.5")
    assert (
        points == calculate_ema([*decimals(range(1, 52)), Decimal(999)], period=50)[:-1]
    )


@pytest.mark.parametrize(
    ("closes", "expected"),
    [
        ([Decimal(1)] * 15, Decimal(50)),
        (decimals(range(1, 16)), Decimal(100)),
        (decimals(range(15, 0, -1)), Decimal(0)),
    ],
)
def test_rsi_edges_and_alignment(closes: list[Decimal], expected: Decimal) -> None:
    points = calculate_rsi(closes, period=14)
    assert [(point.candle_index, point.value) for point in points] == [(14, expected)]


def test_rsi_wilder_recursion_range_and_no_future_influence() -> None:
    closes = [
        Decimal(value) for value in (1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9)
    ]
    points = calculate_rsi(closes, period=14)
    assert all(Decimal(0) <= point.value <= Decimal(100) for point in points)
    assert points == calculate_rsi([*closes, Decimal(100)], period=14)[:-1]


def test_atr_seed_gap_recursion_alignment_and_no_future_influence() -> None:
    highs = [Decimal(11)] * 15
    lows = [Decimal(9)] * 15
    closes = [Decimal(10)] * 14 + [Decimal(20)]
    points = calculate_atr(highs, lows, closes, period=14)
    assert points[0].candle_index == 13
    assert points[0].value == Decimal(2)
    assert points[1].value >= 0
    assert (
        points
        == calculate_atr(
            [*highs, Decimal(30)], [*lows, Decimal(1)], [*closes, Decimal(2)], period=14
        )[:-1]
    )


def test_atr_rejects_invalid_prices() -> None:
    with pytest.raises(ValueError, match="high"):
        calculate_atr([Decimal(1)], [Decimal(2)], [Decimal(1)], period=1)


@pytest.mark.parametrize(("period", "index"), [(1, 1), (12, 12)])
def test_returns_are_fractional_and_aligned(period: int, index: int) -> None:
    closes = [Decimal(100)] * period + [Decimal(101)]
    point = calculate_return(closes, period=period)[0]
    assert (point.candle_index, point.value) == (index, Decimal("0.01"))


def test_returns_reject_zero_and_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="zero"):
        calculate_return([Decimal(0), Decimal(1)], period=1)
    with pytest.raises(ValueError, match="finite"):
        calculate_return([Decimal("NaN")], period=1)


def test_relative_volume_excludes_current_baseline_and_omits_zero_baseline() -> None:
    volumes = [Decimal(2)] * 20 + [Decimal(3)]
    assert calculate_relative_volume(volumes, period=20)[0].value == Decimal("1.5")
    assert calculate_relative_volume([Decimal(0)] * 20 + [Decimal(3)], period=20) == []


@pytest.mark.parametrize("value", [Decimal(-1), Decimal("Infinity")])
def test_relative_volume_rejects_invalid_input(value: Decimal) -> None:
    with pytest.raises(ValueError):
        calculate_relative_volume([value], period=1)
