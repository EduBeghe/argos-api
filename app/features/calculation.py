from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.ema import calculate_ema
from app.features.models import FeatureName, MarketFeature
from app.features.persistence import insert_market_features
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord

EMA_20_PERIOD = 20
EMA_20_VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureCalculationResult:
    asset: Asset
    feature_name: FeatureName
    feature_version: int
    candle_count_read: int
    values_calculated: int
    values_inserted: int
    duplicates: int
    first_feature_time: datetime | None
    last_feature_time: datetime | None

    def __post_init__(self) -> None:
        if self.values_inserted + self.duplicates != self.values_calculated:
            raise ValueError("inserted plus duplicates must equal calculated")


async def calculate_ema_20_features(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    calculated_at: datetime,
) -> FeatureCalculationResult:
    """Calculate and stage every missing EMA 20 feature without committing."""

    statement = (
        select(MarketCandleRecord.open_time, MarketCandleRecord.close_price)
        .where(
            MarketCandleRecord.asset == asset.value,
            MarketCandleRecord.provider == provider.value,
            MarketCandleRecord.market_type == market_type.value,
            MarketCandleRecord.interval == interval.value,
        )
        .order_by(MarketCandleRecord.open_time.asc())
    )
    candles = (await session.execute(statement)).tuples().all()
    values = calculate_ema([close for _, close in candles], period=EMA_20_PERIOD)
    valid_candles = candles[EMA_20_PERIOD - 1 :]
    features = [
        MarketFeature(
            asset=asset,
            provider=provider,
            market_type=market_type,
            interval=interval,
            candle_open_time=open_time,
            feature_name=FeatureName.EMA_20,
            feature_version=EMA_20_VERSION,
            value=value,
            calculated_at=calculated_at,
        )
        for (open_time, _), value in zip(valid_candles, values, strict=True)
    ]
    inserted = await insert_market_features(session, features) if features else 0
    return FeatureCalculationResult(
        asset=asset,
        feature_name=FeatureName.EMA_20,
        feature_version=EMA_20_VERSION,
        candle_count_read=len(candles),
        values_calculated=len(features),
        values_inserted=inserted,
        duplicates=len(features) - inserted,
        first_feature_time=features[0].candle_open_time if features else None,
        last_feature_time=features[-1].candle_open_time if features else None,
    )
