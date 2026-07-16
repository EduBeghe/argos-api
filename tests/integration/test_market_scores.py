from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.features.calculation import calculate_market_features
from app.features.persistence import MarketFeatureRecord
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from app.scores.calculation import calculate_market_scores
from app.scores.persistence import MarketScoreRecord
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2026, 7, 15, tzinfo=UTC)
CALCULATED_AT = datetime(2026, 7, 16, tzinfo=UTC)


async def test_complete_features_generate_idempotent_score() -> None:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        await session.execute(delete(MarketScoreRecord))
        await session.execute(delete(MarketFeatureRecord))
        await session.execute(delete(MarketCandleRecord))
        for index in range(50):
            open_time = START + timedelta(minutes=5 * index)
            session.add(
                to_record(
                    make_candle(
                        asset=Asset.BTC_USDT,
                        open_time=open_time,
                        close_time=open_time + timedelta(minutes=5, milliseconds=-1),
                        open_price=Decimal(100),
                        high_price=Decimal(200),
                        low_price=Decimal(99),
                        close_price=Decimal(100 + index),
                        quote_asset_volume=Decimal(1000 + index),
                    )
                )
            )
        await session.commit()
        await calculate_market_features(
            session=session,
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=CALCULATED_AT,
        )
        await session.commit()
        first = await calculate_market_scores(
            session=session,
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=CALCULATED_AT,
        )
        await session.commit()
        second = await calculate_market_scores(
            session=session,
            asset=Asset.BTC_USDT,
            provider=Provider.BINANCE,
            market_type=MarketType.SPOT,
            interval=Interval.FIVE_MINUTES,
            calculated_at=CALCULATED_AT,
        )
        await session.commit()
        records = (await session.scalars(select(MarketScoreRecord))).all()
        assert (first.scores_inserted, second.duplicates) == (1, 1)
        assert len(records) == 1
        assert records[0].candle_open_time == START + timedelta(minutes=245)
        await session.execute(delete(MarketScoreRecord))
        await session.execute(delete(MarketFeatureRecord))
        await session.execute(delete(MarketCandleRecord))
        await session.commit()
    await engine.dispose()
