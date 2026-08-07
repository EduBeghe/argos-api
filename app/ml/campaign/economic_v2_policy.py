import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

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


def strategy_policy_grid(horizon_seconds: int) -> tuple[StrategyPolicy, ...]:
    policies: list[StrategyPolicy] = []
    for confidence in ECONOMIC_V2_CONFIDENCE_FRACTIONS:
        for multiplier in ECONOMIC_V2_HOLDING_MULTIPLIERS:
            holding_candles = exact_holding_candles(horizon_seconds, multiplier)
            for regime in ECONOMIC_V2_REGIME_POLICIES:
                payload = {
                    "confidence_fraction": str(confidence),
                    "holding_candles": holding_candles,
                    "holding_multiplier": str(multiplier),
                    "regime_policy": regime,
                    "version": ECONOMIC_V2_STRATEGY_POLICY_VERSION,
                }
                fingerprint = hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                policies.append(
                    StrategyPolicy(
                        fingerprint,
                        confidence,
                        multiplier,
                        regime,
                        holding_candles,
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
