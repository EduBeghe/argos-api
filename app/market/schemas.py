from datetime import UTC, datetime
from decimal import Decimal
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.market.catalog import MarketInstrumentDefinition
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.history import HistoryOrder, HistoryWindow
from app.market.models import MarketCandle

CandleOrder = HistoryOrder


class InstrumentHistoryQuery(HistoryWindow):
    """Complete or catalog-resolvable market identity with a history window."""

    asset: Asset
    interval: Interval
    provider: Provider | None = None
    market_type: MarketType | None = None


class MarketCandleQuery(InstrumentHistoryQuery):
    """Validated filters for persisted market candles."""

    start_time: datetime | None = Field(
        default=None, description="Inclusive UTC candle open-time lower bound."
    )
    end_time: datetime | None = Field(
        default=None, description="Inclusive UTC candle open-time upper bound."
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        for field_name in ("start_time", "end_time"):
            value = getattr(self, field_name)
            if value is not None:
                if value.tzinfo is None or value.utcoffset() is None:
                    raise ValueError(f"{field_name} must be timezone-aware")
                setattr(self, field_name, value.astimezone(UTC))
        if self.from_time is not None and self.start_time is not None:
            raise ValueError("use either from or start_time, not both")
        if self.to_time is not None and self.end_time is not None:
            raise ValueError("use either to or end_time, not both")
        lower = self.from_time or self.start_time
        upper = self.to_time or self.end_time
        if lower is not None and upper is not None and upper < lower:
            raise ValueError("upper time bound must not be before lower time bound")
        return self

    @property
    def effective_from(self) -> datetime | None:
        return self.from_time or self.start_time

    @property
    def effective_to(self) -> datetime | None:
        return self.to_time or self.end_time


class ScoreHistoryQuery(InstrumentHistoryQuery):
    """Filters for persisted technical Score history."""


class DecisionHistoryQuery(InstrumentHistoryQuery):
    """Filters for immutable Decision Snapshot history."""

    run_id: UUID | None = None


class MarketCandleResponse(BaseModel):
    """Public representation of one persisted market candle."""

    id: UUID
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    open_time: datetime
    close_time: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    base_asset_volume: Decimal
    quote_asset_volume: Decimal
    trade_count: int
    taker_buy_base_asset_volume: Decimal
    taker_buy_quote_asset_volume: Decimal
    ingested_at: datetime

    @classmethod
    def from_domain(cls, candle: MarketCandle) -> "MarketCandleResponse":
        """Map a domain candle without exposing its persistence record."""

        return cls.model_validate(candle.model_dump(mode="python"))


class MarketStatusResponse(BaseModel):
    """Persistence and freshness status for one supported asset."""

    asset: Asset
    interval: Interval
    candle_count: int
    latest_open_time: datetime | None
    latest_close_time: datetime | None
    latest_ingested_at: datetime | None
    age_seconds: int | None
    is_fresh: bool


class MarketInstrumentCapabilities(BaseModel):
    ingestion: bool
    scoring: bool
    decision: bool
    paper_trading: bool


class MarketInstrumentResponse(BaseModel):
    asset: Asset
    display_symbol: str
    display_name: str
    asset_class: str
    base_currency: str
    quote_currency: str
    provider: Provider
    provider_symbol: str
    market_type: MarketType
    interval: Interval
    capabilities: MarketInstrumentCapabilities
    display_order: int

    @classmethod
    def from_definition(
        cls, definition: MarketInstrumentDefinition
    ) -> "MarketInstrumentResponse":
        return cls(
            asset=definition.asset,
            display_symbol=definition.display_symbol,
            display_name=definition.display_name,
            asset_class=definition.asset_class,
            base_currency=definition.base_currency,
            quote_currency=definition.quote_currency,
            provider=definition.provider,
            provider_symbol=definition.provider_symbol,
            market_type=definition.market_type,
            interval=definition.interval,
            capabilities=MarketInstrumentCapabilities(
                ingestion=definition.ingestion_enabled,
                scoring=definition.scoring_enabled,
                decision=definition.decision_enabled,
                paper_trading=definition.decision_enabled,
            ),
            display_order=definition.display_order,
        )
