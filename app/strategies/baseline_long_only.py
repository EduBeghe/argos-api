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

    def __post_init__(self) -> None:
        if self.strategy_version <= 0 or self.configuration_version <= 0:
            raise ValueError("strategy and configuration versions must be positive")
        if self.required_score_model_version <= 0:
            raise ValueError("required score model version must be positive")
        if self.action not in _EMITTED_ACTIONS:
            raise ValueError("action is not emitted by baseline_long_only version 1")
        if self.action is StrategyAction.OPEN_POSITION:
            valid_sizing = (
                self.requested_target_portfolio_percentage
                == OPEN_POSITION_TARGET_PORTFOLIO_PERCENTAGE
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
    *, overall_score: Decimal, position_is_open: bool
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
        if overall_score >= ENTRY_THRESHOLD:
            return _decision(
                action=StrategyAction.OPEN_POSITION,
                requested_target_portfolio_percentage=(
                    OPEN_POSITION_TARGET_PORTFOLIO_PERCENTAGE
                ),
                applied_rule="entry_threshold_met",
                explanation="Entry threshold met; request a new long position.",
            )
        return _decision(
            action=StrategyAction.NO_ACTION,
            applied_rule="entry_threshold_not_met",
            explanation="Entry threshold not met; leave the portfolio unchanged.",
        )

    if overall_score >= EXIT_THRESHOLD:
        return _decision(
            action=StrategyAction.NO_ACTION,
            applied_rule="open_position_retained",
            explanation=(
                "Exit threshold not breached; leave the open position unchanged."
            ),
        )
    return _decision(
        action=StrategyAction.SELL_ALL,
        requested_position_percentage=SELL_ALL_POSITION_PERCENTAGE,
        applied_rule="exit_threshold_breached",
        explanation="Exit threshold breached; request complete position liquidation.",
    )


def _decision(
    *,
    action: StrategyAction,
    applied_rule: str,
    explanation: str,
    requested_target_portfolio_percentage: Decimal | None = None,
    requested_position_percentage: Decimal | None = None,
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
    )
