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
    seed = sum(decimals(range(1, 51)), start=Decimal(0)) / Decimal(50)
    alpha = Decimal(2) / Decimal(51)
    first_recursive = alpha * Decimal(51) + (Decimal(1) - alpha) * seed
    assert [(point.candle_index, point.value) for point in points] == [
        (49, seed),
        (50, first_recursive),
    ]
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


def test_rsi_mixed_seed_and_wilder_recursion_are_exact() -> None:
    closes = [
        Decimal(value) for value in (1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9)
    ]
    points = calculate_rsi(closes, period=14)

    # First 14 changes: +1, -1, then six repetitions of +2, -1.
    average_gain = Decimal(13) / Decimal(14)
    average_loss = Decimal(7) / Decimal(14)
    seeded_rsi = Decimal(100) - Decimal(100) / (
        Decimal(1) + average_gain / average_loss
    )
    next_average_gain = (average_gain * Decimal(13) + Decimal(2)) / Decimal(14)
    next_average_loss = (average_loss * Decimal(13) + Decimal(0)) / Decimal(14)
    next_rsi = Decimal(100) - Decimal(100) / (
        Decimal(1) + next_average_gain / next_average_loss
    )

    assert [(point.candle_index, point.value) for point in points] == [
        (14, seeded_rsi),
        (15, next_rsi),
    ]
    assert points == calculate_rsi([*closes, Decimal(100)], period=14)[:-1]


def test_atr_seed_gap_recursion_alignment_and_no_future_influence() -> None:
    highs = [Decimal(11), Decimal(15), *([Decimal(11)] * 13)]
    lows = [Decimal(9), Decimal(14), *([Decimal(9)] * 13)]
    closes = [Decimal(10)] * 15
    points = calculate_atr(highs, lows, closes, period=14)

    first_true_range = Decimal(11) - Decimal(9)
    gap_true_range = max(
        Decimal(15) - Decimal(14),
        abs(Decimal(15) - Decimal(10)),
        abs(Decimal(14) - Decimal(10)),
    )
    assert first_true_range == Decimal(2)
    assert gap_true_range == Decimal(5)
    seeded_atr = (
        first_true_range + gap_true_range + Decimal(12) * Decimal(2)
    ) / Decimal(14)
    next_atr = (seeded_atr * Decimal(13) + Decimal(2)) / Decimal(14)
    assert [(point.candle_index, point.value) for point in points] == [
        (13, seeded_atr),
        (14, next_atr),
    ]
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


def test_returns_negative_zero_twelve_period_and_future_isolation() -> None:
    assert calculate_return([Decimal(100), Decimal(80)], period=1)[0].value == Decimal(
        "-0.2"
    )
    assert calculate_return([Decimal(100), Decimal(100)], period=1)[0].value == 0
    closes = [Decimal(100), *([Decimal(999)] * 11), Decimal(125)]
    points = calculate_return(closes, period=12)
    assert [(point.candle_index, point.value) for point in points] == [
        (12, Decimal("0.25"))
    ]
    assert points == calculate_return([*closes, Decimal(1)], period=12)[:-1]


def test_relative_volume_excludes_current_baseline_and_omits_zero_baseline() -> None:
    volumes = [*decimals(range(1, 21)), Decimal(5)]
    prior_average = sum(decimals(range(1, 21)), start=Decimal(0)) / Decimal(20)
    points = calculate_relative_volume(volumes, period=20)
    assert [(point.candle_index, point.value) for point in points] == [
        (20, Decimal(5) / prior_average)
    ]
    assert points[0].value < 1
    assert points == calculate_relative_volume([*volumes, Decimal(999)], period=20)[:-1]
    assert calculate_relative_volume([Decimal(0)] * 20 + [Decimal(3)], period=20) == []


@pytest.mark.parametrize("value", [Decimal(-1), Decimal("Infinity")])
def test_relative_volume_rejects_invalid_input(value: Decimal) -> None:
    with pytest.raises(ValueError):
        calculate_relative_volume([value], period=1)
