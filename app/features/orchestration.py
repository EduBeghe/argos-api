from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.features.calculation import FeatureCalculationResult, calculate_ema_20_features
from app.market.enums import Asset, Interval, MarketType, Provider

FEATURE_ASSETS = (Asset.BTC_USDT, Asset.ETH_USDT)


async def calculate_all_market_features(
    *, session: AsyncSession, calculated_at: datetime
) -> tuple[FeatureCalculationResult, ...]:
    """Stage EMA 20 for both MVP assets without managing the transaction."""

    results: list[FeatureCalculationResult] = []
    for asset in FEATURE_ASSETS:
        results.append(
            await calculate_ema_20_features(
                session=session,
                asset=asset,
                provider=Provider.BINANCE,
                market_type=MarketType.SPOT,
                interval=Interval.FIVE_MINUTES,
                calculated_at=calculated_at,
            )
        )
    return tuple(results)
