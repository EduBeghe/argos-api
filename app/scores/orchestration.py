from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import enabled_scoring_instruments
from app.scores.calculation import ScoreCalculationResult, calculate_market_scores


async def calculate_all_market_scores(
    *,
    session: AsyncSession,
    calculated_at: datetime,
    target_start_time: datetime | None = None,
    target_end_time: datetime | None = None,
) -> tuple[ScoreCalculationResult, ...]:
    results: list[ScoreCalculationResult] = []
    for instrument in enabled_scoring_instruments():
        if target_start_time is None and target_end_time is None:
            result = await calculate_market_scores(
                session=session,
                asset=instrument.asset,
                provider=instrument.provider,
                market_type=instrument.market_type,
                interval=instrument.interval,
                calculated_at=calculated_at,
            )
        else:
            result = await calculate_market_scores(
                session=session,
                asset=instrument.asset,
                provider=instrument.provider,
                market_type=instrument.market_type,
                interval=instrument.interval,
                calculated_at=calculated_at,
                target_start_time=target_start_time,
                target_end_time=target_end_time,
            )
        results.append(result)
    return tuple(results)
