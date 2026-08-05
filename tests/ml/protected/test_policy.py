from decimal import Decimal

from app.ml.protected.policy import (
    FinalValidationPolicyEvidence,
    failed_final_validation_rules,
)


def evidence(**changes: object) -> FinalValidationPolicyEvidence:
    values: dict[str, object] = {
        "complete_finite": True,
        "round_trips": 20,
        "total_net_return": Decimal("0.10"),
        "half_year_returns": (
            Decimal("0.01"),
            Decimal("0.01"),
            Decimal("0.01"),
            Decimal("-0.05"),
        ),
        "maximum_drawdown": Decimal("0.25"),
        "stressed_return_1_5x": Decimal("0.000000000000000001"),
        "stressed_return_2x": Decimal("0.000000000000000001"),
    }
    values.update(changes)
    return FinalValidationPolicyEvidence(**values)  # type: ignore[arg-type]


def test_policy_accepts_inclusive_boundaries() -> None:
    assert failed_final_validation_rules(evidence()) == ()


def test_policy_reports_every_failed_rule() -> None:
    failed = failed_final_validation_rules(
        evidence(
            complete_finite=False,
            round_trips=19,
            total_net_return=Decimal("0.099"),
            half_year_returns=(Decimal("0"),),
            maximum_drawdown=Decimal("0.2500001"),
            stressed_return_1_5x=Decimal("0"),
            stressed_return_2x=Decimal("-0.1"),
        )
    )
    assert set(failed) == {
        "incomplete_or_non_finite_evidence",
        "insufficient_round_trips",
        "total_net_return_below_10_percent",
        "incomplete_half_year_evidence",
        "fewer_than_three_positive_half_years",
        "maximum_drawdown_above_25_percent",
        "non_positive_return_at_1_5x_costs",
        "non_positive_return_at_2x_costs",
    }
