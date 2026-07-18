from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.market.enums import Asset, Interval, MarketType, Provider
from app.paper.models import PaperPositionStatus, PaperTradingRun
from app.paper.persistence import PaperTradeRecord


class PaperAccountResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    status: str
    account_currency: str
    initial_cash: Decimal
    current_cash: Decimal
    configuration_version: int
    fee_rate: Decimal
    slippage_rate: Decimal
    started_at: datetime
    stopped_at: datetime | None
    stop_reason: str | None
    predecessor_run_id: UUID | None

    @classmethod
    def from_domain(cls, run: PaperTradingRun) -> "PaperAccountResponse":
        return cls.model_validate(run)


class PaperPositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    run_id: UUID
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    status: PaperPositionStatus
    quantity: Decimal
    average_entry_price: Decimal
    total_entry_notional: Decimal
    accumulated_entry_fees: Decimal
    opened_at: datetime
    closed_at: datetime | None
    exit_price: Decimal | None
    gross_realized_pnl: Decimal | None
    net_realized_pnl: Decimal | None


class PaperTradeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    run_id: UUID
    decision_snapshot_id: UUID
    position_id: UUID
    side: str
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    observed_market_price: Decimal
    slippage_rate: Decimal
    execution_price: Decimal
    quantity: Decimal
    executed_notional: Decimal
    fee_rate: Decimal
    fee_amount: Decimal
    cash_impact: Decimal
    executed_at: datetime
    configuration_version: int
    equity_before: Decimal | None

    @classmethod
    def from_record(cls, record: PaperTradeRecord) -> "PaperTradeResponse":
        return cls.model_validate(record)


class PaperDecisionCycleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    paper_run_id: UUID
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    strategy_name: str
    strategy_version: int
    configuration_version: int
    scheduled_boundary: datetime
    source_score_id: UUID | None
    decision_snapshot_id: UUID | None
    paper_trade_id: UUID | None
    result: str | None
    started_at: datetime
    completed_at: datetime | None
    status: str


class ValuedPaperPositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    position_id: UUID
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    quantity: Decimal
    average_entry_price: Decimal
    cost_basis: Decimal
    mark_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_return: Decimal | None
    mark_candle_open_time: datetime


class PaperPortfolioResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    run_id: UUID
    status: str
    account_currency: str
    initial_cash: Decimal
    current_cash: Decimal
    positions_market_value: Decimal
    equity: Decimal
    total_pnl: Decimal
    total_return: Decimal | None
    valued_at: datetime
    positions: tuple[ValuedPaperPositionResponse, ...]


class PaperPositionSnapshotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    position_id: UUID
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    quantity: Decimal
    average_entry_price: Decimal
    cost_basis: Decimal
    mark_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_return: Decimal | None
    mark_candle_open_time: datetime


class PaperPortfolioSnapshotResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    run_id: UUID
    scheduled_boundary: datetime
    recorded_at: datetime
    account_currency: str
    initial_cash: Decimal
    current_cash: Decimal
    positions_market_value: Decimal
    equity: Decimal
    total_pnl: Decimal
    total_return: Decimal | None
    configuration_version: int
    positions: list[PaperPositionSnapshotResponse]
