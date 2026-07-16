from dataclasses import replace
from decimal import Decimal

import pytest

from app.scores.calculator import (
    BaselineScoreInput,
    ScoreInterpretation,
    calculate_baseline_score,
    clamp,
    interpretation_for,
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
    expected_momentum = (
        Decimal("0.35")
        + Decimal("0.20") * Decimal("0.5")
        + Decimal("0.45") * Decimal("0.5")
    )
    expected_directional = (
        Decimal("0.55") * Decimal("0.5") + Decimal("0.45") * expected_momentum
    )
    expected_activity = (Decimal(1) - Decimal("0.5")) / Decimal("1.5")
    expected_confirmation = Decimal("0.75") + Decimal("0.25") * expected_activity
    assert result.momentum_score == expected_momentum
    assert result.volume_activity == expected_activity
    assert result.volatility_ratio == Decimal("0.04")
    assert result.directional_score == expected_directional
    assert result.volume_confirmation_factor == expected_confirmation
    assert result.overall_score == expected_directional * expected_confirmation
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
