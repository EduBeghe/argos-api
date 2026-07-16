from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.market.enums import Asset, Interval, MarketType, Provider
from app.scores.calculation import ScoreCalculationResult, calculate_market_scores

SCORE_ASSETS = (Asset.BTC_USDT, Asset.ETH_USDT)


async def calculate_all_market_scores(
    *, session: AsyncSession, calculated_at: datetime
) -> tuple[ScoreCalculationResult, ...]:
    return tuple(
        [
            await calculate_market_scores(
                session=session,
                asset=asset,
                provider=Provider.BINANCE,
                market_type=MarketType.SPOT,
                interval=Interval.FIVE_MINUTES,
                calculated_at=calculated_at,
            )
            for asset in SCORE_ASSETS
        ]
    )
