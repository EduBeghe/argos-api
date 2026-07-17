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

    @classmethod
    def from_domain(cls, run: PaperTradingRun) -> "PaperAccountResponse":
        return cls.model_validate(run)


class PaperPositionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
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

    @classmethod
    def from_record(cls, record: PaperTradeRecord) -> "PaperTradeResponse":
        return cls.model_validate(record)
