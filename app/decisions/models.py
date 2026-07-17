from datetime import datetime, timedelta
from decimal import Decimal
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Asset, Interval, MarketType, Provider
from app.strategies.baseline_long_only import (
    CONFIGURATION_VERSION,
    OPEN_POSITION_TARGET_PORTFOLIO_PERCENTAGE,
    REQUIRED_SCORE_MODEL_NAME,
    REQUIRED_SCORE_MODEL_VERSION,
    SELL_ALL_POSITION_PERCENTAGE,
    STRATEGY_NAME,
    STRATEGY_VERSION,
    StrategyAction,
)


class DecisionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    decision_timestamp: datetime
    source_candle_open_time: datetime
    source_market_price: Decimal
    source_score_id: UUID
    source_score_model_name: str
    source_score_model_version: int
    source_overall_score: Decimal
    strategy_name: str
    strategy_version: int
    configuration_version: int
    action: StrategyAction
    requested_target_portfolio_percentage: Decimal | None
    requested_position_percentage: Decimal | None
    applied_rule: str
    explanation: str
    position_is_open: bool

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        for timestamp in (self.decision_timestamp, self.source_candle_open_time):
            if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
                raise ValueError("timestamps must be timezone-aware UTC")
        for name in (
            self.source_score_model_name,
            self.strategy_name,
            self.applied_rule,
            self.explanation,
        ):
            if not name.strip():
                raise ValueError("names, rules, and explanations must not be empty")
        if (
            self.strategy_name != STRATEGY_NAME
            or self.strategy_version != STRATEGY_VERSION
            or self.configuration_version != CONFIGURATION_VERSION
            or self.source_score_model_name != REQUIRED_SCORE_MODEL_NAME
            or self.source_score_model_version != REQUIRED_SCORE_MODEL_VERSION
        ):
            raise ValueError(
                "snapshot metadata is incompatible with strategy version 1"
            )
        if not self.source_market_price.is_finite() or self.source_market_price <= 0:
            raise ValueError("source market price must be finite and positive")
        if not self.source_overall_score.is_finite() or not Decimal(
            "-1"
        ) <= self.source_overall_score <= Decimal("1"):
            raise ValueError("source overall score must be finite and between -1 and 1")
        if self.action is StrategyAction.OPEN_POSITION:
            valid = (
                not self.position_is_open
                and self.requested_target_portfolio_percentage
                == OPEN_POSITION_TARGET_PORTFOLIO_PERCENTAGE
                and self.requested_position_percentage is None
                and self.applied_rule == "entry_threshold_met"
            )
        elif self.action is StrategyAction.SELL_ALL:
            valid = (
                self.position_is_open
                and self.requested_target_portfolio_percentage is None
                and self.requested_position_percentage == SELL_ALL_POSITION_PERCENTAGE
                and self.applied_rule == "exit_threshold_breached"
            )
        elif self.action is StrategyAction.NO_ACTION:
            valid = (
                self.requested_target_portfolio_percentage is None
                and self.requested_position_percentage is None
                and self.applied_rule
                in {"entry_threshold_not_met", "open_position_retained"}
            )
        else:
            valid = False
        if not valid:
            raise ValueError(
                "action, portfolio state, rule, and sizing are inconsistent"
            )
        return self
