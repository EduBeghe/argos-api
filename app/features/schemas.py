from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel

from app.features.models import MarketFeature
from app.market.enums import Asset, Interval, MarketType, Provider


class FeatureValuesResponse(BaseModel):
    ema_20: Decimal | None = None
    ema_50: Decimal | None = None
    rsi_14: Decimal | None = None
    atr_14: Decimal | None = None
    return_1: Decimal | None = None
    return_12: Decimal | None = None
    relative_volume_20: Decimal | None = None


class MarketFeatureResponse(BaseModel):
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    candle_open_time: datetime
    calculated_at: datetime
    features: FeatureValuesResponse

    @classmethod
    def from_features(cls, values: list[MarketFeature]) -> "MarketFeatureResponse":
        if not values:
            raise ValueError("feature snapshot must not be empty")
        first = values[0]
        return cls(
            asset=first.asset,
            provider=first.provider,
            market_type=first.market_type,
            interval=first.interval,
            candle_open_time=first.candle_open_time,
            calculated_at=max(value.calculated_at for value in values),
            features=FeatureValuesResponse.model_validate(
                {value.feature_name.value: value.value for value in values}
            ),
        )
