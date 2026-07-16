from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel

from app.market.enums import Asset, Interval, MarketType, Provider
from app.scores.calculator import ScoreInterpretation
from app.scores.models import MarketScore


class ScoreContributionsResponse(BaseModel):
    trend: Decimal
    rsi: Decimal
    return_1: Decimal
    return_12: Decimal


class ScoreExplanationsResponse(BaseModel):
    trend: str
    momentum: str
    volume: str
    volatility: str


class MarketScoreResponse(BaseModel):
    asset: Asset
    provider: Provider
    market_type: MarketType
    interval: Interval
    candle_open_time: datetime
    calculated_at: datetime
    model_name: str
    model_version: int
    trend_score: Decimal
    momentum_score: Decimal
    volume_activity: Decimal
    volatility_ratio: Decimal
    directional_score: Decimal
    volume_confirmation_factor: Decimal
    overall_score: Decimal
    signal_strength: Decimal
    interpretation: ScoreInterpretation
    contributions: ScoreContributionsResponse
    explanations: ScoreExplanationsResponse

    @classmethod
    def from_domain(cls, score: MarketScore) -> "MarketScoreResponse":
        trend_relation = (
            "above"
            if score.trend_score > 0
            else "below"
            if score.trend_score < 0
            else "effectively equal to"
        )
        momentum = (
            "positive"
            if score.momentum_score > 0
            else "negative"
            if score.momentum_score < 0
            else "neutral"
        )
        volume = (
            "above"
            if score.volume_activity > Decimal("0.5")
            else "below"
            if score.volume_activity < Decimal("0.2")
            else "near"
        )
        return cls(
            **score.model_dump(
                mode="python",
                exclude={
                    "id",
                    "trend_contribution",
                    "rsi_contribution",
                    "return_1_contribution",
                    "return_12_contribution",
                },
            ),
            contributions=ScoreContributionsResponse(
                trend=score.trend_contribution,
                rsi=score.rsi_contribution,
                return_1=score.return_1_contribution,
                return_12=score.return_12_contribution,
            ),
            explanations=ScoreExplanationsResponse(
                trend=f"EMA 20 is {trend_relation} EMA 50 relative to ATR.",
                momentum=f"RSI and recent returns produce {momentum} momentum.",
                volume=f"Current volume activity is {volume} its recent baseline.",
                volatility=(
                    f"ATR is {(score.volatility_ratio * Decimal(100))}% "
                    "of candle close."
                ),
            ),
        )
