from datetime import datetime
from decimal import Decimal
from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.history import HistoryWindow
from app.paper.evaluation import (
    BenchmarkMethod,
    EvaluationStatus,
    EvaluationUnavailableReason,
)
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


class PaperTradeQuery(HistoryWindow):
    """Backward-compatible filters for persisted Paper Trade history."""

    run_id: UUID | None = None
    asset: Asset | None = None
    interval: Interval | None = None
    provider: Provider | None = None
    market_type: MarketType | None = None
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (self.asset is None) != (self.interval is None):
            raise ValueError("asset and interval must be provided together")
        if self.asset is None and (
            self.provider is not None or self.market_type is not None
        ):
            raise ValueError("provider and market_type require asset and interval")
        return self


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


class PaperPerformanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    run_id: UUID
    status: str
    account_currency: str
    snapshot_count: int
    initial_equity: Decimal
    latest_equity: Decimal
    absolute_pnl: Decimal
    total_return: Decimal | None
    peak_equity: Decimal
    lowest_equity: Decimal
    current_drawdown: Decimal
    max_drawdown: Decimal
    first_snapshot_boundary: datetime | None
    latest_snapshot_boundary: datetime | None
    evaluated_from: datetime
    evaluated_until: datetime


class PaperEvaluationRunResponse(BaseModel):
    run_id: UUID
    status: str
    account_currency: str
    configuration_version: int
    evaluated_from: datetime
    evaluated_until: datetime
    latest_snapshot_id: UUID | None
    snapshot_count: int


class PaperStrategyEvaluationResponse(BaseModel):
    status: EvaluationStatus
    unavailable_reason: EvaluationUnavailableReason | None
    initial_equity: Decimal
    ending_equity: Decimal | None
    absolute_pnl: Decimal | None
    total_return: Decimal | None
    current_drawdown: Decimal | None
    max_drawdown: Decimal | None
    realized_gross_pnl: Decimal | None
    realized_net_pnl: Decimal | None
    unrealized_pnl: Decimal | None
    fees_paid: Decimal | None
    estimated_slippage_cost: Decimal | None
    decision_count: int
    decision_counts_by_action: dict[str, int]
    trade_count: int
    trade_counts_by_side: dict[str, int]
    open_position_count: int
    closed_position_count: int
    winning_position_count: int
    losing_position_count: int
    breakeven_position_count: int
    win_rate: Decimal | None
    average_win: Decimal | None
    average_loss: Decimal | None
    profit_factor: Decimal | None


class BenchmarkInstrumentResponse(BaseModel):
    asset: str
    provider: str
    market_type: str
    interval: str
    status: EvaluationStatus
    unavailable_reason: EvaluationUnavailableReason | None
    allocation: Decimal
    entry_candle_open_time: datetime | None
    entry_observed_price: Decimal | None
    entry_execution_price: Decimal | None
    quantity: Decimal | None
    entry_fee: Decimal | None
    estimated_entry_slippage_cost: Decimal | None
    ending_candle_open_time: datetime | None
    ending_mark_price: Decimal | None
    ending_market_value: Decimal | None
    absolute_pnl: Decimal | None
    total_return: Decimal | None


class PaperBenchmarkEvaluationResponse(BaseModel):
    status: EvaluationStatus
    unavailable_reason: EvaluationUnavailableReason | None
    method: BenchmarkMethod
    instrument_count: int
    initial_equity: Decimal
    ending_asset_value: Decimal | None
    remaining_cash: Decimal | None
    ending_equity: Decimal | None
    absolute_pnl: Decimal | None
    total_return: Decimal | None
    entry_fees: Decimal | None
    entry_slippage_cost: Decimal | None
    instruments: list[BenchmarkInstrumentResponse]


class PaperEvaluationComparisonResponse(BaseModel):
    status: EvaluationStatus
    unavailable_reason: EvaluationUnavailableReason | None
    strategy_excess_return: Decimal | None
    strategy_minus_benchmark_pnl: Decimal | None


class PaperInstrumentEvaluationResponse(BaseModel):
    asset: str
    provider: str
    market_type: str
    interval: str
    decision_count: int
    decision_counts_by_action: dict[str, int]
    trade_count: int
    trade_counts_by_side: dict[str, int]
    fees_paid: Decimal
    estimated_slippage_cost: Decimal
    open_position_count: int
    closed_position_count: int
    realized_gross_pnl: Decimal
    realized_net_pnl: Decimal
    unrealized_pnl: Decimal | None


class PaperEvaluationResponse(BaseModel):
    run: PaperEvaluationRunResponse
    strategy: PaperStrategyEvaluationResponse
    benchmark: PaperBenchmarkEvaluationResponse
    comparison: PaperEvaluationComparisonResponse
    instruments: list[PaperInstrumentEvaluationResponse]
