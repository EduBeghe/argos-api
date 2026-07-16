from decimal import Decimal

import pytest

from app.features.ema import calculate_ema


def test_fewer_than_period_closes_produces_no_values() -> None:
    assert calculate_ema([Decimal(index) for index in range(1, 20)], period=20) == []


def test_twenty_closes_seed_ema_with_sma() -> None:
    values = calculate_ema([Decimal(index) for index in range(1, 21)], period=20)
    assert [(point.candle_index, point.value) for point in values] == [
        (19, Decimal("10.5"))
    ]


def test_next_close_uses_decimal_recurrence() -> None:
    closes = [Decimal(index) for index in range(1, 22)]
    values = calculate_ema(closes, period=20)
    expected = Decimal(2) / Decimal(21) * Decimal(21) + Decimal(19) / Decimal(
        21
    ) * Decimal("10.5")
    assert [(point.candle_index, point.value) for point in values] == [
        (19, Decimal("10.5")),
        (20, expected),
    ]
    assert all(isinstance(point.value, Decimal) for point in values)


def test_future_close_does_not_change_prior_values() -> None:
    closes = [Decimal(index) for index in range(1, 22)]
    assert (
        calculate_ema(closes, period=20)
        == calculate_ema([*closes, Decimal("999")], period=20)[:-1]
    )


@pytest.mark.parametrize("period", [0, -1])
def test_invalid_period_is_rejected(period: int) -> None:
    with pytest.raises(ValueError, match="period"):
        calculate_ema([Decimal(1)], period=period)


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity")])
def test_non_finite_close_is_rejected(value: Decimal) -> None:
    with pytest.raises(ValueError, match="finite"):
        calculate_ema([value], period=1)
