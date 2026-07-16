from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel

from app.features.models import FeatureName, MarketFeature
from app.market.enums import Asset, Interval, MarketType, Provider


class MarketFeatureResponse(BaseModel):
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    candle_open_time: datetime
    feature_name: FeatureName
    feature_version: int
    value: Decimal
    calculated_at: datetime

    @classmethod
    def from_domain(cls, feature: MarketFeature) -> "MarketFeatureResponse":
        return cls.model_validate(feature.model_dump(mode="python", exclude={"id"}))
