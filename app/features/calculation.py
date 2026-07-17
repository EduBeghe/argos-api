from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.indicators import (
    CalculationPoint,
    calculate_atr,
    calculate_ema,
    calculate_relative_volume,
    calculate_return,
    calculate_rsi,
)
from app.features.models import FeatureName, MarketFeature
from app.features.persistence import (
    get_existing_feature_identities,
    insert_market_features,
)
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord

FEATURE_VERSION = 1
EMA_20_VERSION = FEATURE_VERSION


@dataclass(frozen=True, slots=True)
class PerFeatureCalculationResult:
    feature_name: FeatureName
    feature_version: int
    values_calculated: int
    values_inserted: int
    duplicates: int
    first_feature_time: datetime | None
    last_feature_time: datetime | None

    def __post_init__(self) -> None:
        if self.values_inserted + self.duplicates != self.values_calculated:
            raise ValueError("inserted plus duplicates must equal calculated")


@dataclass(frozen=True, slots=True)
class FeatureCalculationResult:
    asset: Asset
    candle_count_read: int
    values_calculated: int
    values_inserted: int
    duplicates: int
    first_feature_time: datetime | None
    last_feature_time: datetime | None
    features: tuple[PerFeatureCalculationResult, ...]

    def __post_init__(self) -> None:
        if self.values_inserted + self.duplicates != self.values_calculated:
            raise ValueError("inserted plus duplicates must equal calculated")
        if self.values_calculated != sum(
            item.values_calculated for item in self.features
        ):
            raise ValueError("aggregate calculated total must equal feature totals")
        if self.values_inserted != sum(item.values_inserted for item in self.features):
            raise ValueError("aggregate inserted total must equal feature totals")


async def calculate_market_features(
    *,
    session: AsyncSession,
    asset: Asset,
    provider: Provider,
    market_type: MarketType,
    interval: Interval,
    calculated_at: datetime,
) -> FeatureCalculationResult:
    statement = (
        select(
            MarketCandleRecord.open_time,
            MarketCandleRecord.high_price,
            MarketCandleRecord.low_price,
            MarketCandleRecord.close_price,
            MarketCandleRecord.quote_asset_volume,
        )
        .where(
            MarketCandleRecord.asset == asset.value,
            MarketCandleRecord.provider == provider.value,
            MarketCandleRecord.market_type == market_type.value,
            MarketCandleRecord.interval == interval.value,
        )
        .order_by(MarketCandleRecord.open_time.asc())
    )
    candles = (await session.execute(statement)).tuples().all()
    times = [row[0] for row in candles]
    highs = [row[1] for row in candles]
    lows = [row[2] for row in candles]
    closes = [row[3] for row in candles]
    volumes = [row[4] for row in candles]
    points: tuple[tuple[FeatureName, list[CalculationPoint]], ...] = (
        (FeatureName.EMA_20, calculate_ema(closes, period=20)),
        (FeatureName.EMA_50, calculate_ema(closes, period=50)),
        (FeatureName.RSI_14, calculate_rsi(closes, period=14)),
        (FeatureName.ATR_14, calculate_atr(highs, lows, closes, period=14)),
        (FeatureName.RETURN_1, calculate_return(closes, period=1)),
        (FeatureName.RETURN_12, calculate_return(closes, period=12)),
        (FeatureName.RELATIVE_VOLUME_20, calculate_relative_volume(volumes, period=20)),
    )
    values = [
        MarketFeature(
            asset=asset,
            provider=provider,
            market_type=market_type,
            interval=interval,
            candle_open_time=times[point.candle_index],
            feature_name=name,
            feature_version=FEATURE_VERSION,
            value=point.value,
            calculated_at=calculated_at,
        )
        for name, calculated_points in points
        for point in calculated_points
    ]
    existing_identities = (
        await get_existing_feature_identities(
            session=session,
            asset=asset,
            provider=provider,
            market_type=market_type,
            interval=interval,
            feature_version=FEATURE_VERSION,
        )
        if values
        else set()
    )
    missing_values = [
        value
        for value in values
        if (
            value.candle_open_time,
            value.feature_name,
            value.feature_version,
        )
        not in existing_identities
    ]
    inserted_names = (
        await insert_market_features(session, missing_values) if missing_values else []
    )
    inserted_counts = Counter(inserted_names)
    per_feature = tuple(
        PerFeatureCalculationResult(
            feature_name=name,
            feature_version=FEATURE_VERSION,
            values_calculated=len(calculated_points),
            values_inserted=inserted_counts[name],
            duplicates=len(calculated_points) - inserted_counts[name],
            first_feature_time=times[calculated_points[0].candle_index]
            if calculated_points
            else None,
            last_feature_time=times[calculated_points[-1].candle_index]
            if calculated_points
            else None,
        )
        for name, calculated_points in points
    )
    feature_times = [value.candle_open_time for value in values]
    return FeatureCalculationResult(
        asset=asset,
        candle_count_read=len(candles),
        values_calculated=len(values),
        values_inserted=len(inserted_names),
        duplicates=len(values) - len(inserted_names),
        first_feature_time=min(feature_times) if feature_times else None,
        last_feature_time=max(feature_times) if feature_times else None,
        features=per_feature,
    )


calculate_ema_20_features = calculate_market_features


def feature_log_totals(
    results: tuple[FeatureCalculationResult, ...],
) -> list[dict[str, str | int]]:
    return [
        {
            "feature_name": name.value,
            "calculated": sum(
                feature.values_calculated
                for result in results
                for feature in result.features
                if feature.feature_name is name
            ),
            "inserted": sum(
                feature.values_inserted
                for result in results
                for feature in result.features
                if feature.feature_name is name
            ),
            "duplicates": sum(
                feature.duplicates
                for result in results
                for feature in result.features
                if feature.feature_name is name
            ),
        }
        for name in FeatureName
    ]
