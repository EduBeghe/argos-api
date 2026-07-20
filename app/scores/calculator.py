from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum

ZERO = Decimal(0)
ONE = Decimal(1)
SCORE_QUANTUM = Decimal("0.000000000000000001")


@dataclass(frozen=True, slots=True)
class BaselineTechnicalParameters:
    trend_weight: Decimal
    rsi_weight: Decimal
    return_1_weight: Decimal
    return_12_weight: Decimal
    volume_confirmation_weight: Decimal

    def __post_init__(self) -> None:
        directional = (
            self.trend_weight,
            self.rsi_weight,
            self.return_1_weight,
            self.return_12_weight,
        )
        values = (*directional, self.volume_confirmation_weight)
        if any(
            not isinstance(value, Decimal) or not value.is_finite() for value in values
        ):
            raise ValueError("score weights must be finite Decimals")
        if any(not ZERO <= value <= ONE for value in directional):
            raise ValueError("directional weights must be between 0 and 1")
        if sum(directional, ZERO) != ONE:
            raise ValueError("directional weights must sum exactly to 1")
        if not ZERO <= self.volume_confirmation_weight <= Decimal("0.50"):
            raise ValueError("volume confirmation weight must be between 0 and 0.50")

    @property
    def confirmation_base(self) -> Decimal:
        return ONE - self.volume_confirmation_weight


BASELINE_TECHNICAL_V1_PARAMETERS = BaselineTechnicalParameters(
    Decimal("0.55"),
    Decimal("0.1575"),
    Decimal("0.09"),
    Decimal("0.2025"),
    Decimal("0.25"),
)


def quantize_score(value: Decimal) -> Decimal:
    """Round persisted values to scale 18 using decimal half-even."""

    return value.quantize(SCORE_QUANTUM, rounding=ROUND_HALF_EVEN)


class ScoreInterpretation(StrEnum):
    STRONGLY_BEARISH = "strongly_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    STRONGLY_BULLISH = "strongly_bullish"


@dataclass(frozen=True, slots=True)
class BaselineScoreInput:
    close_price: Decimal
    ema_20: Decimal
    ema_50: Decimal
    rsi_14: Decimal
    atr_14: Decimal
    return_1: Decimal
    return_12: Decimal
    relative_volume_20: Decimal


@dataclass(frozen=True, slots=True)
class BaselineScoreOutput:
    trend_score: Decimal
    rsi_signal: Decimal
    return_1_signal: Decimal
    return_12_signal: Decimal
    momentum_score: Decimal
    volume_activity: Decimal
    volatility_ratio: Decimal
    directional_score: Decimal
    volume_confirmation_factor: Decimal
    overall_score: Decimal
    signal_strength: Decimal
    trend_contribution: Decimal
    rsi_contribution: Decimal
    return_1_contribution: Decimal
    return_12_contribution: Decimal
    interpretation: ScoreInterpretation


def clamp(value: Decimal, minimum: Decimal, maximum: Decimal) -> Decimal:
    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    return min(max(value, minimum), maximum)


def interpretation_for(score: Decimal) -> ScoreInterpretation:
    if score <= Decimal("-0.60"):
        return ScoreInterpretation.STRONGLY_BEARISH
    if score <= Decimal("-0.20"):
        return ScoreInterpretation.BEARISH
    if score < Decimal("0.20"):
        return ScoreInterpretation.NEUTRAL
    if score < Decimal("0.60"):
        return ScoreInterpretation.BULLISH
    return ScoreInterpretation.STRONGLY_BULLISH


def calculate_baseline_score(
    value: BaselineScoreInput,
    parameters: BaselineTechnicalParameters = BASELINE_TECHNICAL_V1_PARAMETERS,
) -> BaselineScoreOutput:
    fields = (
        value.__dict__
        if hasattr(value, "__dict__")
        else {name: getattr(value, name) for name in value.__dataclass_fields__}
    )
    if any(not item.is_finite() for item in fields.values()):
        raise ValueError("score inputs must be finite")
    if value.close_price <= 0:
        raise ValueError("close price must be positive")
    if value.atr_14 < 0:
        raise ValueError("ATR must be non-negative")
    if not ZERO <= value.rsi_14 <= Decimal(100):
        raise ValueError("RSI must be between 0 and 100")
    if value.relative_volume_20 < 0:
        raise ValueError("relative volume must be non-negative")
    spread = value.ema_20 - value.ema_50
    if value.atr_14 == 0:
        if spread != 0:
            raise ValueError("non-zero EMA spread is inconsistent with zero ATR")
        trend = ZERO
    else:
        trend = clamp(spread / value.atr_14, -ONE, ONE)
    trend = quantize_score(trend)
    rsi = quantize_score(clamp((value.rsi_14 - Decimal(50)) / Decimal(20), -ONE, ONE))
    return_1 = quantize_score(clamp(value.return_1 / Decimal("0.0025"), -ONE, ONE))
    return_12 = quantize_score(clamp(value.return_12 / Decimal("0.01"), -ONE, ONE))
    momentum_weight = ONE - parameters.trend_weight
    momentum = (
        quantize_score(
            (
                parameters.rsi_weight * rsi
                + parameters.return_1_weight * return_1
                + parameters.return_12_weight * return_12
            )
            / momentum_weight
        )
        if momentum_weight > 0
        else ZERO
    )
    volume = quantize_score(
        clamp(
            (value.relative_volume_20 - Decimal("0.5")) / Decimal("1.5"),
            ZERO,
            ONE,
        )
    )
    volatility = quantize_score(value.atr_14 / value.close_price)
    trend_contribution = quantize_score(parameters.trend_weight * trend)
    rsi_contribution = quantize_score(parameters.rsi_weight * rsi)
    return_1_contribution = quantize_score(parameters.return_1_weight * return_1)
    return_12_contribution = quantize_score(parameters.return_12_weight * return_12)
    directional = quantize_score(
        trend_contribution
        + rsi_contribution
        + return_1_contribution
        + return_12_contribution
    )
    confirmation = quantize_score(
        parameters.confirmation_base + parameters.volume_confirmation_weight * volume
    )
    overall = quantize_score(directional * confirmation)
    return BaselineScoreOutput(
        trend,
        rsi,
        return_1,
        return_12,
        momentum,
        volume,
        volatility,
        directional,
        confirmation,
        overall,
        abs(overall),
        trend_contribution,
        rsi_contribution,
        return_1_contribution,
        return_12_contribution,
        interpretation_for(overall),
    )
