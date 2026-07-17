from dataclasses import replace
from decimal import Decimal

import pytest

from app.scores.calculator import (
    BaselineScoreInput,
    ScoreInterpretation,
    calculate_baseline_score,
    clamp,
    interpretation_for,
    quantize_score,
)


def inputs() -> BaselineScoreInput:
    return BaselineScoreInput(
        close_price=Decimal(100),
        ema_20=Decimal(102),
        ema_50=Decimal(100),
        rsi_14=Decimal(70),
        atr_14=Decimal(4),
        return_1=Decimal("0.00125"),
        return_12=Decimal("0.005"),
        relative_volume_20=Decimal(1),
    )


def test_clamp_preserves_decimal_boundaries() -> None:
    assert clamp(Decimal(2), Decimal(-1), Decimal(1)) == Decimal(1)
    assert clamp(Decimal("0.2"), Decimal(-1), Decimal(1)) == Decimal("0.2")


def test_exact_baseline_formula_and_contributions() -> None:
    result = calculate_baseline_score(inputs())
    assert result.trend_score == Decimal("0.5")
    assert result.rsi_signal == 1
    assert result.return_1_signal == Decimal("0.5")
    assert result.return_12_signal == Decimal("0.5")
    expected_momentum = quantize_score(
        Decimal("0.35")
        + Decimal("0.20") * Decimal("0.5")
        + Decimal("0.45") * Decimal("0.5")
    )
    expected_activity = quantize_score((Decimal(1) - Decimal("0.5")) / Decimal("1.5"))
    expected_confirmation = quantize_score(
        Decimal("0.75") + Decimal("0.25") * expected_activity
    )
    assert result.momentum_score == expected_momentum
    assert result.volume_activity == expected_activity
    assert result.volatility_ratio == Decimal("0.04")
    expected_directional = quantize_score(
        result.trend_contribution
        + result.rsi_contribution
        + result.return_1_contribution
        + result.return_12_contribution
    )
    assert result.directional_score == expected_directional
    assert result.volume_confirmation_factor == expected_confirmation
    assert result.overall_score == quantize_score(
        expected_directional * expected_confirmation
    )
    assert result.signal_strength == abs(result.overall_score)
    assert (
        result.trend_contribution
        + result.rsi_contribution
        + result.return_1_contribution
        + result.return_12_contribution
        == result.directional_score
    )


@pytest.mark.parametrize(("rsi", "expected"), [(30, -1), (50, 0), (70, 1)])
def test_rsi_normalization(rsi: int, expected: int) -> None:
    assert (
        calculate_baseline_score(replace(inputs(), rsi_14=Decimal(rsi))).rsi_signal
        == expected
    )


def test_saturation_volume_and_zero_atr_edges() -> None:
    assert (
        calculate_baseline_score(replace(inputs(), return_1=Decimal(1))).return_1_signal
        == 1
    )
    assert (
        calculate_baseline_score(
            replace(inputs(), return_12=Decimal(-1))
        ).return_12_signal
        == -1
    )
    assert (
        calculate_baseline_score(
            replace(inputs(), relative_volume_20=Decimal("0.5"))
        ).volume_activity
        == 0
    )
    assert (
        calculate_baseline_score(
            replace(inputs(), relative_volume_20=Decimal(2))
        ).volume_activity
        == 1
    )
    equal = replace(
        inputs(), atr_14=Decimal(0), ema_20=Decimal(100), ema_50=Decimal(100)
    )
    assert calculate_baseline_score(equal).trend_score == 0
    with pytest.raises(ValueError, match="inconsistent"):
        calculate_baseline_score(replace(inputs(), atr_14=Decimal(0)))


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        ("-0.60", ScoreInterpretation.STRONGLY_BEARISH),
        ("-0.20", ScoreInterpretation.BEARISH),
        ("0.199", ScoreInterpretation.NEUTRAL),
        ("0.20", ScoreInterpretation.BULLISH),
        ("0.60", ScoreInterpretation.STRONGLY_BULLISH),
    ],
)
def test_interpretation_boundaries(score: str, expected: ScoreInterpretation) -> None:
    assert interpretation_for(Decimal(score)) is expected


@pytest.mark.parametrize(
    "change",
    [
        {"rsi_14": Decimal(101)},
        {"relative_volume_20": Decimal(-1)},
        {"close_price": Decimal(0)},
        {"atr_14": Decimal(-1)},
        {"ema_20": Decimal("NaN")},
    ],
)
def test_invalid_inputs_are_rejected(change: dict[str, Decimal]) -> None:
    with pytest.raises(ValueError):
        calculate_baseline_score(replace(inputs(), **change))


def test_repeated_execution_is_deterministic_and_volume_cannot_reverse_sign() -> None:
    low = calculate_baseline_score(replace(inputs(), relative_volume_20=Decimal(0)))
    high = calculate_baseline_score(replace(inputs(), relative_volume_20=Decimal(2)))
    assert calculate_baseline_score(inputs()) == calculate_baseline_score(inputs())
    assert low.overall_score > 0 and high.overall_score > 0
    assert abs(low.overall_score) <= abs(high.overall_score)


def test_half_even_quantization_is_symmetric_and_persisted_invariants_hold() -> None:
    assert quantize_score(Decimal("0.1234567890123456785")) == Decimal(
        "0.123456789012345678"
    )
    assert quantize_score(Decimal("-0.1234567890123456785")) == Decimal(
        "-0.123456789012345678"
    )
    assert quantize_score(Decimal("0.9999999999999999995")) == Decimal(1)
    assert quantize_score(Decimal("-0.9999999999999999995")) == Decimal(-1)
    value = calculate_baseline_score(
        replace(
            inputs(),
            ema_20=Decimal("100.1234567890123456789"),
            rsi_14=Decimal("63.333333333333333333"),
            return_1=Decimal("0.001234567890123456789"),
            return_12=Decimal("-0.003333333333333333333"),
            relative_volume_20=Decimal("1.111111111111111111"),
        )
    )
    persisted = (
        value.trend_score,
        value.momentum_score,
        value.volume_activity,
        value.volatility_ratio,
        value.directional_score,
        value.volume_confirmation_factor,
        value.overall_score,
        value.signal_strength,
        value.trend_contribution,
        value.rsi_contribution,
        value.return_1_contribution,
        value.return_12_contribution,
    )
    exponents = [item.as_tuple().exponent for item in persisted]
    assert all(isinstance(exponent, int) and exponent >= -18 for exponent in exponents)
    assert (
        value.trend_contribution
        + value.rsi_contribution
        + value.return_1_contribution
        + value.return_12_contribution
        == value.directional_score
    )
    assert value.signal_strength == abs(value.overall_score)
    assert value.interpretation is interpretation_for(value.overall_score)
    assert value == calculate_baseline_score(
        replace(
            inputs(),
            ema_20=Decimal("100.1234567890123456789"),
            rsi_14=Decimal("63.333333333333333333"),
            return_1=Decimal("0.001234567890123456789"),
            return_12=Decimal("-0.003333333333333333333"),
            relative_volume_20=Decimal("1.111111111111111111"),
        )
    )
