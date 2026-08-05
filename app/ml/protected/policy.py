from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class FinalValidationPolicyEvidence:
    complete_finite: bool
    round_trips: int
    total_net_return: Decimal
    half_year_returns: tuple[Decimal, ...]
    maximum_drawdown: Decimal
    stressed_return_1_5x: Decimal
    stressed_return_2x: Decimal


def failed_final_validation_rules(
    evidence: FinalValidationPolicyEvidence,
) -> tuple[str, ...]:
    failed: list[str] = []
    gates = (
        (evidence.complete_finite, "incomplete_or_non_finite_evidence"),
        (evidence.round_trips >= 20, "insufficient_round_trips"),
        (
            evidence.total_net_return >= Decimal("0.10"),
            "total_net_return_below_10_percent",
        ),
        (len(evidence.half_year_returns) == 4, "incomplete_half_year_evidence"),
        (
            sum(value > 0 for value in evidence.half_year_returns) >= 3,
            "fewer_than_three_positive_half_years",
        ),
        (
            all(value >= Decimal("-0.05") for value in evidence.half_year_returns),
            "half_year_loss_below_minus_5_percent",
        ),
        (
            evidence.maximum_drawdown <= Decimal("0.25"),
            "maximum_drawdown_above_25_percent",
        ),
        (evidence.stressed_return_1_5x > 0, "non_positive_return_at_1_5x_costs"),
        (evidence.stressed_return_2x > 0, "non_positive_return_at_2x_costs"),
    )
    failed.extend(code for passed, code in gates if not passed)
    return tuple(failed)
