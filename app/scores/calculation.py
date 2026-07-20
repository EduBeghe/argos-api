from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.models import FeatureName
from app.features.persistence import MarketFeatureRecord
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord
from app.scores.calculator import BaselineScoreInput, calculate_baseline_score
from app.scores.models import MarketScore
from app.scores.persistence import MarketScoreRecord, insert_market_scores

MODEL_NAME = "baseline_technical"
MODEL_VERSION = 1
REQUIRED_FEATURES = (
    FeatureName.EMA_20,
    FeatureName.EMA_50,
    FeatureName.RSI_14,
    FeatureName.ATR_14,
    FeatureName.RETURN_1,
    FeatureName.RETURN_12,
    FeatureName.RELATIVE_VOLUME_20,
)


@dataclass(frozen=True, slots=True)
class ScoreCalculationResult:
    asset: Asset
    model_name: str
    model_version: int
    candidate_candles_read: int
    complete_feature_sets: int
    scores_calculated: int
    scores_inserted: int
    duplicates: int
    first_score_time: datetime | None
    last_score_time: datetime | None

    def __post_init__(self) -> None:
        if self.scores_calculated != self.complete_feature_sets:
            raise ValueError("calculated scores must equal complete feature sets")
        if self.scores_inserted + self.duplicates != self.scores_calculated:
            raise ValueError("inserted plus duplicates must equal calculated")


async def calculate_market_scores(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    calculated_at: datetime,
    target_start_time: datetime | None = None,
    target_end_time: datetime | None = None,
) -> ScoreCalculationResult:
    statement = (
        select(
            MarketCandleRecord.open_time,
            MarketCandleRecord.close_price,
            MarketFeatureRecord.feature_name,
            MarketFeatureRecord.value,
        )
        .join(
            MarketFeatureRecord,
            (MarketFeatureRecord.asset == MarketCandleRecord.asset)
            & (MarketFeatureRecord.provider == MarketCandleRecord.provider)
            & (MarketFeatureRecord.market_type == MarketCandleRecord.market_type)
            & (MarketFeatureRecord.interval == MarketCandleRecord.interval)
            & (MarketFeatureRecord.candle_open_time == MarketCandleRecord.open_time),
        )
        .where(
            MarketCandleRecord.asset == asset.value,
            MarketCandleRecord.provider == provider.value,
            MarketCandleRecord.market_type == market_type.value,
            MarketCandleRecord.interval == interval.value,
            MarketFeatureRecord.feature_version == 1,
            MarketFeatureRecord.feature_name.in_(
                [name.value for name in REQUIRED_FEATURES]
            ),
            ~exists().where(
                MarketScoreRecord.asset == MarketCandleRecord.asset,
                MarketScoreRecord.provider == MarketCandleRecord.provider,
                MarketScoreRecord.market_type == MarketCandleRecord.market_type,
                MarketScoreRecord.interval == MarketCandleRecord.interval,
                MarketScoreRecord.candle_open_time == MarketCandleRecord.open_time,
                MarketScoreRecord.model_name == MODEL_NAME,
                MarketScoreRecord.model_version == MODEL_VERSION,
            ),
        )
        .order_by(MarketCandleRecord.open_time.asc())
    )
    if target_start_time is not None:
        statement = statement.where(MarketCandleRecord.open_time >= target_start_time)
    if target_end_time is not None:
        statement = statement.where(MarketCandleRecord.open_time < target_end_time)
    rows = (await session.execute(statement)).tuples().all()
    grouped: dict[datetime, tuple[Decimal, dict[FeatureName, Decimal]]] = {}
    for open_time, close_price, feature_name, feature_value in rows:
        current = grouped.setdefault(open_time, (close_price, {}))
        current[1][FeatureName(feature_name)] = feature_value
    scores: list[MarketScore] = []
    for open_time, (close_price, features) in grouped.items():
        if set(features) != set(REQUIRED_FEATURES):
            continue
        output = calculate_baseline_score(
            BaselineScoreInput(
                close_price=close_price,
                ema_20=features[FeatureName.EMA_20],
                ema_50=features[FeatureName.EMA_50],
                rsi_14=features[FeatureName.RSI_14],
                atr_14=features[FeatureName.ATR_14],
                return_1=features[FeatureName.RETURN_1],
                return_12=features[FeatureName.RETURN_12],
                relative_volume_20=features[FeatureName.RELATIVE_VOLUME_20],
            )
        )
        scores.append(
            MarketScore(
                asset=asset,
                provider=provider,
                market_type=market_type,
                interval=interval,
                candle_open_time=open_time,
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                calculated_at=calculated_at,
                trend_score=output.trend_score,
                momentum_score=output.momentum_score,
                volume_activity=output.volume_activity,
                volatility_ratio=output.volatility_ratio,
                directional_score=output.directional_score,
                volume_confirmation_factor=output.volume_confirmation_factor,
                overall_score=output.overall_score,
                signal_strength=output.signal_strength,
                trend_contribution=output.trend_contribution,
                rsi_contribution=output.rsi_contribution,
                return_1_contribution=output.return_1_contribution,
                return_12_contribution=output.return_12_contribution,
                interpretation=output.interpretation,
            )
        )
    inserted = await insert_market_scores(session, scores) if scores else 0
    return ScoreCalculationResult(
        asset,
        MODEL_NAME,
        MODEL_VERSION,
        len(grouped),
        len(scores),
        len(scores),
        inserted,
        len(scores) - inserted,
        scores[0].candle_open_time if scores else None,
        scores[-1].candle_open_time if scores else None,
    )
