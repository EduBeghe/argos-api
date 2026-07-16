from datetime import datetime, timedelta
from decimal import Decimal
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.market.enums import Asset, Interval, MarketType, Provider
from app.scores.calculator import ScoreInterpretation, interpretation_for


class MarketScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    candle_open_time: datetime
    model_name: str
    model_version: int = Field(gt=0)
    calculated_at: datetime
    trend_score: Decimal = Field(ge=-1, le=1)
    momentum_score: Decimal = Field(ge=-1, le=1)
    volume_activity: Decimal = Field(ge=0, le=1)
    volatility_ratio: Decimal = Field(ge=0)
    directional_score: Decimal = Field(ge=-1, le=1)
    volume_confirmation_factor: Decimal = Field(ge=Decimal("0.75"), le=1)
    overall_score: Decimal = Field(ge=-1, le=1)
    signal_strength: Decimal = Field(ge=0, le=1)
    trend_contribution: Decimal
    rsi_contribution: Decimal
    return_1_contribution: Decimal
    return_12_contribution: Decimal
    interpretation: ScoreInterpretation

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        if self.model_name != "baseline_technical":
            raise ValueError("unsupported score model")
        for timestamp in (self.candle_open_time, self.calculated_at):
            if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
                raise ValueError("timestamps must be timezone-aware UTC")
        decimal_values = [
            value for name, value in self.__dict__.items() if isinstance(value, Decimal)
        ]
        if any(not value.is_finite() for value in decimal_values):
            raise ValueError("score values must be finite")
        if self.signal_strength != abs(self.overall_score):
            raise ValueError("signal strength must equal absolute overall score")
        contributions = (
            self.trend_contribution
            + self.rsi_contribution
            + self.return_1_contribution
            + self.return_12_contribution
        )
        if contributions != self.directional_score:
            raise ValueError("contributions must sum to directional score")
        if self.interpretation is not interpretation_for(self.overall_score):
            raise ValueError("interpretation must match overall score")
        return self
