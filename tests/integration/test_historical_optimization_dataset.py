from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.features.persistence import MarketFeatureRecord
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from app.optimization.feature_matrix import build_historical_feature_matrix
from app.optimization.historical_data import (
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
    read_historical_dataset,
)
from app.optimization.refinement import refine_historical_dataset
from app.optimization.simulation import simulate_historical_strategy
from app.paper.persistence import PaperPositionRecord, PaperTradeRecord
from app.scores.persistence import MarketScoreRecord
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2024, 3, 1, tzinfo=UTC)
COUNT = 60
END = START + timedelta(minutes=5 * COUNT)
INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as value:
        cleanup = delete(MarketCandleRecord).where(
            MarketCandleRecord.open_time >= START, MarketCandleRecord.open_time < END
        )
        await value.execute(cleanup)
        await value.commit()
        for index in range(COUNT):
            opened = START + timedelta(minutes=5 * index)
            value.add(
                to_record(
                    make_candle(
                        asset=Asset.BTC_USDT,
                        open_time=opened,
                        close_time=opened + timedelta(minutes=5, milliseconds=-1),
                        open_price=Decimal(index + 2),
                        high_price=Decimal(index + 3),
                        low_price=Decimal(index + 1),
                        close_price=Decimal(index + 2),
                        quote_asset_volume=Decimal(100 + index),
                    )
                )
            )
        await value.commit()
        yield value
        await value.rollback()
        await value.execute(cleanup)
        await value.commit()
    await engine.dispose()


async def test_keyset_multi_page_matrix_is_read_only(session: AsyncSession) -> None:
    feature_count_before = await session.scalar(
        select(func.count()).select_from(MarketFeatureRecord)
    )
    score_count_before = await session.scalar(
        select(func.count()).select_from(MarketScoreRecord)
    )
    dataset = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    matrix = build_historical_feature_matrix(dataset)
    assert dataset.actual_candle_count == COUNT
    assert len(matrix.rows) == COUNT - 49
    assert (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord))
        == feature_count_before
    )
    assert (
        await session.scalar(select(func.count()).select_from(MarketScoreRecord))
        == score_count_before
    )
    assert not session.new and not session.dirty and not session.deleted


async def test_historical_simulation_is_read_only(session: AsyncSession) -> None:
    counts_before = (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord)),
        await session.scalar(select(func.count()).select_from(MarketScoreRecord)),
        await session.scalar(select(func.count()).select_from(PaperTradeRecord)),
        await session.scalar(select(func.count()).select_from(PaperPositionRecord)),
    )
    dataset = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    result = simulate_historical_strategy(
        dataset=dataset,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
    )
    counts_after = (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord)),
        await session.scalar(select(func.count()).select_from(MarketScoreRecord)),
        await session.scalar(select(func.count()).select_from(PaperTradeRecord)),
        await session.scalar(select(func.count()).select_from(PaperPositionRecord)),
    )
    assert result.decision_count > 0
    assert counts_after == counts_before
    assert not session.new and not session.dirty and not session.deleted


async def test_historical_refinement_is_read_only(session: AsyncSession) -> None:
    counts_before = (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord)),
        await session.scalar(select(func.count()).select_from(MarketScoreRecord)),
        await session.scalar(select(func.count()).select_from(PaperTradeRecord)),
        await session.scalar(select(func.count()).select_from(PaperPositionRecord)),
    )
    dataset = await read_historical_dataset(
        session=session,
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, END)
        ),
        now=END + timedelta(days=1),
        page_size=17,
    )
    result = refine_historical_dataset(
        dataset=dataset,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        validation_top_k=3,
    )
    counts_after = (
        await session.scalar(select(func.count()).select_from(MarketFeatureRecord)),
        await session.scalar(select(func.count()).select_from(MarketScoreRecord)),
        await session.scalar(select(func.count()).select_from(PaperTradeRecord)),
        await session.scalar(select(func.count()).select_from(PaperPositionRecord)),
    )
    assert result.total_candidate_count == 60
    assert counts_after == counts_before
    assert not session.new and not session.dirty and not session.deleted
