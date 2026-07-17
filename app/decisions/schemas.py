from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from app.decisions.models import DecisionSnapshot
from app.market.enums import Asset, Interval, MarketType, Provider
from app.strategies.baseline_long_only import StrategyAction


class DecisionSnapshotResponse(BaseModel):
    id: UUID
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

    @classmethod
    def from_domain(cls, snapshot: DecisionSnapshot) -> "DecisionSnapshotResponse":
        return cls.model_validate(snapshot.model_dump(mode="python"))
