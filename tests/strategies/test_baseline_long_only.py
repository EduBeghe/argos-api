from decimal import Decimal

import pytest

from app.strategies.baseline_long_only import (
    CONFIGURATION_VERSION,
    OPEN_POSITION_TARGET_PORTFOLIO_PERCENTAGE,
    REQUIRED_SCORE_MODEL_NAME,
    REQUIRED_SCORE_MODEL_VERSION,
    SELL_ALL_POSITION_PERCENTAGE,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    StrategyAction,
    StrategyDecision,
    evaluate_baseline_long_only,
)


@pytest.mark.parametrize(
    ("score", "position_is_open", "action", "applied_rule"),
    [
        (Decimal("0.60"), False, StrategyAction.OPEN_POSITION, "entry_threshold_met"),
        (
            Decimal("0.599999999999999999"),
            False,
            StrategyAction.NO_ACTION,
            "entry_threshold_not_met",
        ),
        (
            Decimal("-1"),
            False,
            StrategyAction.NO_ACTION,
            "entry_threshold_not_met",
        ),
        (
            Decimal("0.90"),
            True,
            StrategyAction.NO_ACTION,
            "open_position_retained",
        ),
        (
            Decimal("0.20"),
            True,
            StrategyAction.NO_ACTION,
            "open_position_retained",
        ),
        (
            Decimal("0.199999999999999999"),
            True,
            StrategyAction.SELL_ALL,
            "exit_threshold_breached",
        ),
        (
            Decimal("-1"),
            True,
            StrategyAction.SELL_ALL,
            "exit_threshold_breached",
        ),
    ],
)
def test_policy_boundaries_and_rules(
    score: Decimal,
    position_is_open: bool,
    action: StrategyAction,
    applied_rule: str,
) -> None:
    result = evaluate_baseline_long_only(
        overall_score=score, position_is_open=position_is_open
    )

    assert result.action is action
    assert result.applied_rule == applied_rule


@pytest.mark.parametrize(
    ("score", "position_is_open"),
    [
        (Decimal("-1"), False),
        (Decimal("0"), False),
        (Decimal("0.60"), False),
        (Decimal("1"), False),
        (Decimal("-1"), True),
        (Decimal("0"), True),
        (Decimal("0.20"), True),
        (Decimal("1"), True),
    ],
)
def test_version_one_never_emits_future_actions(
    score: Decimal, position_is_open: bool
) -> None:
    result = evaluate_baseline_long_only(
        overall_score=score, position_is_open=position_is_open
    )

    assert result.action not in {
        StrategyAction.INCREASE_POSITION,
        StrategyAction.SELL_PARTIAL,
    }


def test_action_sizing_and_metadata() -> None:
    opened = evaluate_baseline_long_only(
        overall_score=Decimal("0.60"), position_is_open=False
    )
    retained = evaluate_baseline_long_only(
        overall_score=Decimal("0.20"), position_is_open=True
    )
    sold = evaluate_baseline_long_only(
        overall_score=Decimal("0.19"), position_is_open=True
    )

    assert opened.requested_target_portfolio_percentage == Decimal("15.00")
    assert opened.requested_position_percentage is None
    assert retained.requested_target_portfolio_percentage is None
    assert retained.requested_position_percentage is None
    assert sold.requested_target_portfolio_percentage is None
    assert sold.requested_position_percentage == Decimal("100.00")
    assert Decimal("15.00") == OPEN_POSITION_TARGET_PORTFOLIO_PERCENTAGE
    assert Decimal("100.00") == SELL_ALL_POSITION_PERCENTAGE
    assert (
        opened.strategy_name,
        opened.strategy_version,
        opened.configuration_version,
        opened.required_score_model_name,
        opened.required_score_model_version,
    ) == (
        STRATEGY_NAME,
        STRATEGY_VERSION,
        CONFIGURATION_VERSION,
        REQUIRED_SCORE_MODEL_NAME,
        REQUIRED_SCORE_MODEL_VERSION,
    )


def test_repeated_evaluation_is_deterministic() -> None:
    first = evaluate_baseline_long_only(
        overall_score=Decimal("0.72"), position_is_open=False
    )
    second = evaluate_baseline_long_only(
        overall_score=Decimal("0.72"), position_is_open=False
    )

    assert first == second


@pytest.mark.parametrize(
    "score",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("-1.000000000000000001"),
        Decimal("1.000000000000000001"),
    ],
)
def test_invalid_decimal_scores_are_rejected(score: Decimal) -> None:
    with pytest.raises(ValueError):
        evaluate_baseline_long_only(overall_score=score, position_is_open=False)


def test_float_is_rejected_without_conversion() -> None:
    with pytest.raises(ValueError, match="Decimal"):
        evaluate_baseline_long_only(overall_score=0.60, position_is_open=False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("action", "target_percentage", "position_percentage"),
    [
        (StrategyAction.OPEN_POSITION, Decimal("14.99"), None),
        (StrategyAction.SELL_ALL, None, Decimal("99.99")),
        (StrategyAction.NO_ACTION, Decimal("15.00"), None),
        (StrategyAction.INCREASE_POSITION, None, None),
        (StrategyAction.SELL_PARTIAL, None, Decimal("50.00")),
    ],
)
def test_result_rejects_invalid_action_or_sizing(
    action: StrategyAction,
    target_percentage: Decimal | None,
    position_percentage: Decimal | None,
) -> None:
    with pytest.raises(ValueError):
        StrategyDecision(
            action=action,
            requested_target_portfolio_percentage=target_percentage,
            requested_position_percentage=position_percentage,
            strategy_name=STRATEGY_NAME,
            strategy_version=STRATEGY_VERSION,
            configuration_version=CONFIGURATION_VERSION,
            required_score_model_name=REQUIRED_SCORE_MODEL_NAME,
            required_score_model_version=REQUIRED_SCORE_MODEL_VERSION,
            applied_rule="invalid",
            explanation="Invalid test result.",
        )
