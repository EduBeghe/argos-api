from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.features.calculation import FeatureCalculationResult, calculate_market_features
from app.market.catalog import enabled_scoring_instruments


async def calculate_all_market_features(
    *, session: AsyncSession, calculated_at: datetime
) -> tuple[FeatureCalculationResult, ...]:
    """Stage EMA 20 for both MVP assets without managing the transaction."""

    results: list[FeatureCalculationResult] = []
    for instrument in enabled_scoring_instruments():
        results.append(
            await calculate_market_features(
                session=session,
                asset=instrument.asset,
                provider=instrument.provider,
                market_type=instrument.market_type,
                interval=instrument.interval,
                calculated_at=calculated_at,
            )
        )
    return tuple(results)
