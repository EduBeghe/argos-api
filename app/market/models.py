from decimal import Decimal
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, Field, model_validator

from app.market.enums import Asset, Interval, MarketType, Provider

PositiveDecimal = Annotated[Decimal, Field(gt=0)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0)]
NonNegativeInteger = Annotated[int, Field(ge=0)]


class MarketCandle(BaseModel):
    """Provider-independent normalized market candle."""

    id: UUID = Field(default_factory=uuid4)
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    open_time: AwareDatetime
    close_time: AwareDatetime
    open_price: PositiveDecimal
    high_price: PositiveDecimal
    low_price: PositiveDecimal
    close_price: PositiveDecimal
    base_asset_volume: NonNegativeDecimal
    quote_asset_volume: NonNegativeDecimal
    trade_count: NonNegativeInteger
    taker_buy_base_asset_volume: NonNegativeDecimal
    taker_buy_quote_asset_volume: NonNegativeDecimal
    ingested_at: AwareDatetime

    @model_validator(mode="after")
    def validate_candle(self) -> Self:
        """Validate temporal and OHLC relationships."""

        if self.close_time < self.open_time:
            raise ValueError("close_time must not be before open_time")
        if self.high_price < max(self.open_price, self.close_price, self.low_price):
            raise ValueError("high_price must be the greatest OHLC price")
        if self.low_price > min(self.open_price, self.close_price, self.high_price):
            raise ValueError("low_price must be the lowest OHLC price")
        return self
