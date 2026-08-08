from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from app.ml.refinement.contracts import EconomicPolicySearchConfiguration
from app.ml.refinement.economic_policy_contract import economic_policy_fingerprint

from .contracts import (
    ECONOMIC_V2_CONFIDENCE_FRACTIONS,
    ECONOMIC_V2_HOLDING_MULTIPLIERS,
    ECONOMIC_V2_QUALIFICATION_END,
    ECONOMIC_V2_REGIME_POLICIES,
    ECONOMIC_V2_STRATEGY_POLICY_VERSION,
)


@dataclass(frozen=True, slots=True)
class StrategyPolicy:
    fingerprint: str
    confidence_fraction: Decimal
    holding_multiplier: Decimal
    regime_policy: str
    holding_candles: int
    version: int = ECONOMIC_V2_STRATEGY_POLICY_VERSION


def exact_holding_candles(horizon_seconds: int, multiplier: Decimal) -> int:
    candles = Decimal(horizon_seconds) * multiplier / Decimal(300)
    integral = candles.to_integral_value()
    if candles != integral or integral <= 0:
        raise ValueError("holding multiplier must map to exact five-minute candles")
    return int(integral)


def strategy_policy_grid(
    horizon_seconds: int,
    configuration: EconomicPolicySearchConfiguration | None = None,
) -> tuple[StrategyPolicy, ...]:
    if configuration is None:
        configuration = EconomicPolicySearchConfiguration(
            confidence_fractions=ECONOMIC_V2_CONFIDENCE_FRACTIONS,
            holding_multipliers=ECONOMIC_V2_HOLDING_MULTIPLIERS,
            regime_policies=ECONOMIC_V2_REGIME_POLICIES,
            version=ECONOMIC_V2_STRATEGY_POLICY_VERSION,
        )
    policies: list[StrategyPolicy] = []
    for confidence in sorted(configuration.confidence_fractions):
        for multiplier in sorted(configuration.holding_multipliers):
            holding_candles = exact_holding_candles(horizon_seconds, multiplier)
            for regime in sorted(configuration.regime_policies):
                fingerprint = economic_policy_fingerprint(
                    confidence_fraction=confidence,
                    holding_multiplier=multiplier,
                    regime_policy=regime,
                    holding_candles=holding_candles,
                    version=configuration.version,
                )
                policies.append(
                    StrategyPolicy(
                        fingerprint,
                        confidence,
                        multiplier,
                        regime,
                        holding_candles,
                        configuration.version,
                    )
                )
    return tuple(policies)


def enforce_v2_market_boundary(*, start: datetime, end: datetime) -> None:
    boundary = datetime.fromisoformat(ECONOMIC_V2_QUALIFICATION_END).astimezone(UTC)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("v2 market-data boundaries must be timezone-aware")
    if not start < end <= boundary:
        raise ValueError("economic_policy_search_v2 cannot read 2025 evidence")


def failed_qualification_rules(
    *,
    round_trips: int,
    base_return: Decimal,
    return_1_5x: Decimal,
    return_2x: Decimal,
    maximum_drawdown: Decimal,
    quarter_returns: tuple[Decimal, Decimal, Decimal, Decimal],
) -> tuple[str, ...]:
    failures: list[str] = []
    if round_trips < 20:
        failures.append("minimum_round_trips")
    if base_return < Decimal("0.05"):
        failures.append("minimum_base_return")
    if return_1_5x <= 0:
        failures.append("positive_1_5x_cost_return")
    if return_2x <= 0:
        failures.append("positive_2x_cost_return")
    if maximum_drawdown > Decimal("0.20"):
        failures.append("maximum_drawdown")
    if sum(value > 0 for value in quarter_returns) < 3:
        failures.append("positive_quarters")
    if min(quarter_returns) < Decimal("-0.05"):
        failures.append("worst_quarter")
    return tuple(failures)
