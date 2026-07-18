from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import enabled_scoring_instruments
from app.scores.calculation import ScoreCalculationResult, calculate_market_scores


async def calculate_all_market_scores(
    *, session: AsyncSession, calculated_at: datetime
) -> tuple[ScoreCalculationResult, ...]:
    return tuple(
        [
            await calculate_market_scores(
                session=session,
                asset=instrument.asset,
                provider=instrument.provider,
                market_type=instrument.market_type,
                interval=instrument.interval,
                calculated_at=calculated_at,
            )
            for instrument in enabled_scoring_instruments()
        ]
    )
