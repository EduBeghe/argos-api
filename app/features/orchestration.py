from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.features.calculation import FeatureCalculationResult, calculate_market_features
from app.market.catalog import enabled_scoring_instruments


async def calculate_all_market_features(
    *,
    session: AsyncSession,
    calculated_at: datetime,
    target_start_time: datetime | None = None,
    target_end_time: datetime | None = None,
    context_limit: int | None = None,
) -> tuple[FeatureCalculationResult, ...]:
    """Stage EMA 20 for both MVP assets without managing the transaction."""

    results: list[FeatureCalculationResult] = []
    for instrument in enabled_scoring_instruments():
        if (
            target_start_time is None
            and target_end_time is None
            and context_limit is None
        ):
            result = await calculate_market_features(
                session=session,
                asset=instrument.asset,
                provider=instrument.provider,
                market_type=instrument.market_type,
                interval=instrument.interval,
                calculated_at=calculated_at,
            )
        else:
            result = await calculate_market_features(
                session=session,
                asset=instrument.asset,
                provider=instrument.provider,
                market_type=instrument.market_type,
                interval=instrument.interval,
                calculated_at=calculated_at,
                target_start_time=target_start_time,
                target_end_time=target_end_time,
                context_limit=context_limit,
            )
        results.append(result)
    return tuple(results)
