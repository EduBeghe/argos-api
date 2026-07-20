from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

STRATEGY_NAME = "baseline_long_only"
STRATEGY_VERSION = 1
CONFIGURATION_VERSION = 1
REQUIRED_SCORE_MODEL_NAME = "baseline_technical"
REQUIRED_SCORE_MODEL_VERSION = 1
ENTRY_THRESHOLD = Decimal("0.60")
EXIT_THRESHOLD = Decimal("0.20")
OPEN_POSITION_TARGET_PORTFOLIO_PERCENTAGE = Decimal("15.00")
SELL_ALL_POSITION_PERCENTAGE = Decimal("100.00")

_MINIMUM_SCORE = Decimal("-1")
_MAXIMUM_SCORE = Decimal("1")


@dataclass(frozen=True, slots=True)
class BaselineLongOnlyParameters:
    entry_threshold: Decimal
    exit_threshold: Decimal
    open_position_target_percentage: Decimal

    def __post_init__(self) -> None:
        values = (
            self.entry_threshold,
            self.exit_threshold,
            self.open_position_target_percentage,
        )
        if any(
            not isinstance(value, Decimal) or not value.is_finite() for value in values
        ):
            raise ValueError("strategy parameters must be finite Decimals")
        if not (
            _MINIMUM_SCORE <= self.entry_threshold <= _MAXIMUM_SCORE
            and _MINIMUM_SCORE <= self.exit_threshold <= _MAXIMUM_SCORE
        ):
            raise ValueError("strategy thresholds must be between -1 and 1")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit threshold must be lower than entry threshold")
        if not Decimal(0) < self.open_position_target_percentage <= Decimal(100):
            raise ValueError("target percentage must be greater than 0 and at most 100")


BASELINE_LONG_ONLY_V1_PARAMETERS = BaselineLongOnlyParameters(
    ENTRY_THRESHOLD,
    EXIT_THRESHOLD,
    OPEN_POSITION_TARGET_PORTFOLIO_PERCENTAGE,
)


class StrategyAction(StrEnum):
    OPEN_POSITION = "open_position"
    INCREASE_POSITION = "increase_position"
    SELL_PARTIAL = "sell_partial"
    SELL_ALL = "sell_all"
    NO_ACTION = "no_action"


_EMITTED_ACTIONS = frozenset(
    {
        StrategyAction.OPEN_POSITION,
        StrategyAction.SELL_ALL,
        StrategyAction.NO_ACTION,
    }
)
_APPLIED_RULES_BY_ACTION = {
    StrategyAction.OPEN_POSITION: frozenset({"entry_threshold_met"}),
    StrategyAction.SELL_ALL: frozenset({"exit_threshold_breached"}),
    StrategyAction.NO_ACTION: frozenset(
        {"entry_threshold_not_met", "open_position_retained"}
    ),
}


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    action: StrategyAction
    requested_target_portfolio_percentage: Decimal | None
    requested_position_percentage: Decimal | None
    strategy_name: str
    strategy_version: int
    configuration_version: int
    required_score_model_name: str
    required_score_model_version: int
    applied_rule: str
    explanation: str
    parameters: BaselineLongOnlyParameters = BASELINE_LONG_ONLY_V1_PARAMETERS

    def __post_init__(self) -> None:
        if (
            self.strategy_name != STRATEGY_NAME
            or self.strategy_version != STRATEGY_VERSION
            or self.configuration_version != CONFIGURATION_VERSION
            or self.required_score_model_name != REQUIRED_SCORE_MODEL_NAME
            or self.required_score_model_version != REQUIRED_SCORE_MODEL_VERSION
        ):
            raise ValueError("metadata must match baseline_long_only version 1")
        if self.action not in _EMITTED_ACTIONS:
            raise ValueError("action is not emitted by baseline_long_only version 1")
        if not isinstance(self.applied_rule, str) or not self.applied_rule.strip():
            raise ValueError("applied_rule must be a non-empty string")
        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("explanation must be a non-empty string")
        if self.applied_rule not in _APPLIED_RULES_BY_ACTION[self.action]:
            raise ValueError("applied_rule is inconsistent with action")
        if self.action is StrategyAction.OPEN_POSITION:
            valid_sizing = (
                self.requested_target_portfolio_percentage
                == self.parameters.open_position_target_percentage
                and self.requested_position_percentage is None
            )
        elif self.action is StrategyAction.SELL_ALL:
            valid_sizing = (
                self.requested_target_portfolio_percentage is None
                and self.requested_position_percentage == SELL_ALL_POSITION_PERCENTAGE
            )
        else:
            valid_sizing = (
                self.requested_target_portfolio_percentage is None
                and self.requested_position_percentage is None
            )
        if not valid_sizing:
            raise ValueError("requested sizing is inconsistent with action")


def evaluate_baseline_long_only(
    *,
    overall_score: Decimal,
    position_is_open: bool,
    parameters: BaselineLongOnlyParameters = BASELINE_LONG_ONLY_V1_PARAMETERS,
) -> StrategyDecision:
    if not isinstance(overall_score, Decimal):
        raise ValueError("overall_score must be a Decimal")
    if not overall_score.is_finite():
        raise ValueError("overall_score must be finite")
    if not _MINIMUM_SCORE <= overall_score <= _MAXIMUM_SCORE:
        raise ValueError("overall_score must be between -1 and 1")
    if not isinstance(position_is_open, bool):
        raise ValueError("position_is_open must be a bool")

    if not position_is_open:
        if overall_score >= parameters.entry_threshold:
            return _decision(
                action=StrategyAction.OPEN_POSITION,
                requested_target_portfolio_percentage=(
                    parameters.open_position_target_percentage
                ),
                applied_rule="entry_threshold_met",
                explanation="Entry threshold met; request a new long position.",
                parameters=parameters,
            )
        return _decision(
            action=StrategyAction.NO_ACTION,
            applied_rule="entry_threshold_not_met",
            explanation="Entry threshold not met; leave the portfolio unchanged.",
            parameters=parameters,
        )

    if overall_score >= parameters.exit_threshold:
        return _decision(
            action=StrategyAction.NO_ACTION,
            applied_rule="open_position_retained",
            explanation=(
                "Exit threshold not breached; leave the open position unchanged."
            ),
            parameters=parameters,
        )
    return _decision(
        action=StrategyAction.SELL_ALL,
        requested_position_percentage=SELL_ALL_POSITION_PERCENTAGE,
        applied_rule="exit_threshold_breached",
        explanation="Exit threshold breached; request complete position liquidation.",
        parameters=parameters,
    )


def _decision(
    *,
    action: StrategyAction,
    applied_rule: str,
    explanation: str,
    requested_target_portfolio_percentage: Decimal | None = None,
    requested_position_percentage: Decimal | None = None,
    parameters: BaselineLongOnlyParameters = BASELINE_LONG_ONLY_V1_PARAMETERS,
) -> StrategyDecision:
    return StrategyDecision(
        action=action,
        requested_target_portfolio_percentage=requested_target_portfolio_percentage,
        requested_position_percentage=requested_position_percentage,
        strategy_name=STRATEGY_NAME,
        strategy_version=STRATEGY_VERSION,
        configuration_version=CONFIGURATION_VERSION,
        required_score_model_name=REQUIRED_SCORE_MODEL_NAME,
        required_score_model_version=REQUIRED_SCORE_MODEL_VERSION,
        applied_rule=applied_rule,
        explanation=explanation,
        parameters=parameters,
    )
