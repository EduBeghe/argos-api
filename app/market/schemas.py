from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.market.catalog import MarketInstrumentDefinition
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.models import MarketCandle


class CandleOrder(StrEnum):
    """Supported chronological candle orderings."""

    ASC = "asc"
    DESC = "desc"


class MarketCandleQuery(BaseModel):
    """Validated filters for persisted market candles."""

    asset: Asset = Field(description="Market asset to return.")
    interval: Interval = Field(description="Market candle interval to return.")
    start_time: datetime | None = Field(
        default=None, description="Inclusive UTC candle open-time lower bound."
    )
    end_time: datetime | None = Field(
        default=None, description="Inclusive UTC candle open-time upper bound."
    )
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum rows.")
    order: CandleOrder = Field(
        default=CandleOrder.DESC, description="Chronological result order."
    )

    @model_validator(mode="after")
    def validate_time_range(self) -> Self:
        for field_name, value in (
            ("start_time", self.start_time),
            ("end_time", self.end_time),
        ):
            if value is not None and (
                value.tzinfo is None or value.utcoffset() != timedelta(0)
            ):
                raise ValueError(f"{field_name} must be a timezone-aware UTC datetime")
        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise ValueError("end_time must not be before start_time")
        return self


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
