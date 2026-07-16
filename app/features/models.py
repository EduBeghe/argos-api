from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Asset, Interval, MarketType, Provider


class FeatureName(StrEnum):
    EMA_20 = "ema_20"
    EMA_50 = "ema_50"
    RSI_14 = "rsi_14"
    ATR_14 = "atr_14"
    RETURN_1 = "return_1"
    RETURN_12 = "return_12"
    RELATIVE_VOLUME_20 = "relative_volume_20"


class MarketFeature(BaseModel):
    """One immutable, versioned feature value associated with a candle."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    candle_open_time: datetime
    feature_name: FeatureName
    feature_version: int = Field(gt=0)
    value: Decimal
    calculated_at: datetime

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        for field_name, value in (
            ("candle_open_time", self.candle_open_time),
            ("calculated_at", self.calculated_at),
        ):
            if value.tzinfo is None or value.utcoffset() != timedelta(0):
                raise ValueError(f"{field_name} must be timezone-aware UTC")
        if not self.value.is_finite():
            raise ValueError("value must be finite")
        return self
