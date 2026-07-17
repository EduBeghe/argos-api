from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.features.calculation import FeatureCalculationResult, calculate_ema_20_features
from app.features.models import FeatureName
from app.features.persistence import MarketFeatureRecord
from app.jobs.calculate_market_features import run as run_feature_job
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord, to_record
from tests.market.factories import make_candle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
START = datetime(2026, 7, 15, 10, tzinfo=UTC)
CALCULATED_AT = datetime(2026, 7, 16, tzinfo=UTC)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as database_session:
        await database_session.execute(delete(MarketFeatureRecord))
        await database_session.execute(delete(MarketCandleRecord))
        await database_session.commit()
        yield database_session
        await database_session.rollback()
        await database_session.execute(delete(MarketFeatureRecord))
        await database_session.execute(delete(MarketCandleRecord))
        await database_session.commit()
    await engine.dispose()


async def add_candles(
    session: AsyncSession, count: int, asset: Asset = Asset.BTC_USDT
) -> None:
    existing = len(
        (
            await session.scalars(
                select(MarketCandleRecord.id).where(
                    MarketCandleRecord.asset == asset.value
                )
            )
        ).all()
    )
    for index in range(existing, count):
        open_time = START + timedelta(minutes=5 * index)
        session.add(
            to_record(
                make_candle(
                    asset=asset,
                    open_time=open_time,
                    close_time=open_time + timedelta(minutes=5, milliseconds=-1),
                    close_price=Decimal(index + 1),
                    high_price=Decimal(max(index + 1, 2)),
                    low_price=Decimal(1),
                    open_price=Decimal(1),
                )
            )
        )
    await session.commit()


async def calculate(session: AsyncSession) -> FeatureCalculationResult:
    return await calculate_ema_20_features(
        session=session,
        asset=Asset.BTC_USDT,
        provider=Provider.BINANCE,
        market_type=MarketType.SPOT,
        interval=Interval.FIVE_MINUTES,
        calculated_at=CALCULATED_AT,
    )


async def test_ema_feature_history_is_seeded_incremental_and_idempotent(
    session: AsyncSession,
) -> None:
    await add_candles(session, 19)
    insufficient = await calculate(session)
    await session.commit()
    assert insufficient.values_calculated == 36

    await add_candles(session, 20)
    seeded = await calculate(session)
    await session.commit()
    assert (seeded.values_calculated, seeded.values_inserted) == (41, 5)

    repeated = await calculate(session)
    await session.commit()
    assert (repeated.values_inserted, repeated.duplicates) == (0, 41)

    await add_candles(session, 21)
    incremental = await calculate(session)
    await session.commit()
    assert (incremental.values_calculated, incremental.values_inserted) == (47, 6)

    records = (
        await session.scalars(
            select(MarketFeatureRecord)
            .where(MarketFeatureRecord.feature_name == "ema_20")
            .order_by(MarketFeatureRecord.candle_open_time)
        )
    ).all()
    assert len(records) == 2
    assert records[0].value == Decimal("10.5")
    assert records[0].candle_open_time == START + timedelta(minutes=95)


async def committed_feature_assets() -> list[str]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as verification_session:
        assets = list(
            await verification_session.scalars(
                select(MarketFeatureRecord.asset)
                .distinct()
                .order_by(MarketFeatureRecord.asset)
            )
        )
    await engine.dispose()
    return assets


async def test_feature_job_commits_btc_and_eth_atomically(
    session: AsyncSession,
) -> None:
    await add_candles(session, 20, Asset.BTC_USDT)
    await add_candles(session, 20, Asset.ETH_USDT)

    results = await run_feature_job(
        settings=get_settings(), clock=lambda: CALCULATED_AT
    )

    assert [result.asset for result in results] == [Asset.BTC_USDT, Asset.ETH_USDT]
    assert await committed_feature_assets() == ["BTC/USDT", "ETH/USDT"]


async def test_fifty_candles_persist_complete_feature_pack(
    session: AsyncSession,
) -> None:
    await add_candles(session, 50, Asset.BTC_USDT)

    calculated = await calculate(session)
    await session.commit()
    names = set(
        await session.scalars(
            select(MarketFeatureRecord.feature_name).where(
                MarketFeatureRecord.asset == Asset.BTC_USDT.value
            )
        )
    )

    assert names == {
        "ema_20",
        "ema_50",
        "rsi_14",
        "atr_14",
        "return_1",
        "return_12",
        "relative_volume_20",
    }
    assert calculated.values_inserted == calculated.values_calculated


async def test_large_reconstruction_is_batched_incremental_and_gap_safe(
    session: AsyncSession,
) -> None:
    await add_candles(session, 500, Asset.BTC_USDT)

    first = await calculate(session)
    await session.commit()
    assert first.values_calculated > 3276
    assert first.values_inserted == first.values_calculated

    repeated = await calculate(session)
    await session.commit()
    assert repeated.values_inserted == 0
    assert repeated.duplicates == repeated.values_calculated

    historical = await session.scalar(
        select(MarketFeatureRecord)
        .where(MarketFeatureRecord.feature_name == FeatureName.RETURN_1.value)
        .order_by(MarketFeatureRecord.candle_open_time.asc())
        .limit(1)
    )
    assert historical is not None
    historical_identity = (
        historical.candle_open_time,
        historical.feature_name,
        historical.value,
    )
    await session.delete(historical)
    await session.commit()

    rebuilt = await calculate(session)
    await session.commit()
    assert rebuilt.values_inserted == 1
    reconstructed = await session.scalar(
        select(MarketFeatureRecord).where(
            MarketFeatureRecord.candle_open_time == historical_identity[0],
            MarketFeatureRecord.feature_name == historical_identity[1],
        )
    )
    assert reconstructed is not None
    assert reconstructed.value == historical_identity[2]


async def test_feature_job_rolls_back_btc_when_eth_fails(session: AsyncSession) -> None:
    await add_candles(session, 20, Asset.BTC_USDT)
    await add_candles(session, 20, Asset.ETH_USDT)
    original = RuntimeError("ETH calculation failed")
    real_calculation = calculate_ema_20_features

    async def fail_after_btc(
        *,
        session: AsyncSession,
        asset: Asset,
        provider: Provider,
        market_type: MarketType,
        interval: Interval,
        calculated_at: datetime,
    ) -> FeatureCalculationResult:
        if asset is Asset.ETH_USDT:
            raise original
        return await real_calculation(
            session=session,
            asset=asset,
            provider=provider,
            market_type=market_type,
            interval=interval,
            calculated_at=calculated_at,
        )

    with (
        patch(
            "app.features.orchestration.calculate_market_features",
            fail_after_btc,
        ),
        pytest.raises(RuntimeError) as captured,
    ):
        await run_feature_job(settings=get_settings(), clock=lambda: CALCULATED_AT)

    assert captured.value is original
    assert await committed_feature_assets() == []
