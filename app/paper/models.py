from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from app.market.enums import Asset, Interval, MarketType, Provider


class PaperRunStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class PaperPositionStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class PaperTradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class PaperTradingRun:
    id: UUID
    status: PaperRunStatus
    account_currency: str
    initial_cash: Decimal
    current_cash: Decimal
    configuration_version: int
    fee_rate: Decimal
    slippage_rate: Decimal
    started_at: datetime
    ended_at: datetime | None
    reset_reason: str | None
    predecessor_run_id: UUID | None


@dataclass(frozen=True, slots=True)
class PaperPosition:
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


@dataclass(frozen=True, slots=True)
class PaperTrade:
    id: UUID
    run_id: UUID
    decision_snapshot_id: UUID
    position_id: UUID
    side: PaperTradeSide
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


@dataclass(frozen=True, slots=True)
class ExecutionCalculation:
    side: PaperTradeSide
    execution_price: Decimal
    quantity: Decimal
    executed_notional: Decimal
    fee_amount: Decimal
    cash_impact: Decimal
    resulting_cash: Decimal
    gross_realized_pnl: Decimal | None = None
    net_realized_pnl: Decimal | None = None


@dataclass(frozen=True, slots=True)
class NoTrade:
    reason: str = "decision requested no action"
