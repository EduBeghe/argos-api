from datetime import UTC, datetime
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.features.calculation import FeatureCalculationResult, calculate_ema_20_features
from app.features.persistence import MarketFeatureRecord
from app.jobs.ingest_closed_candles import run
from app.market.enums import Asset, Interval, MarketType, Provider
from app.market.persistence import MarketCandleRecord

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 7, 15, 12, 7, 30, tzinfo=UTC)
OPEN_TIME = 1_784_116_800_000


def raw_kline() -> list[object]:
    return [
        OPEN_TIME,
        "35000.00",
        "35100.00",
        "34900.00",
        "35050.00",
        "12.50",
        OPEN_TIME + 299_999,
        "438125.00",
        100,
        "6.25",
        "219062.50",
        "0",
    ]


def raw_klines(count: int) -> list[list[object]]:
    rows: list[list[object]] = []
    for index in range(count):
        row = raw_kline()
        open_time = OPEN_TIME + index * 300_000
        row[0] = open_time
        row[6] = open_time + 299_999
        row[4] = str(35_000 + index)
        rows.append(row)
    return rows


async def test_job_commits_both_assets_without_real_binance() -> None:
    database_url = get_settings().database_url
    cleanup_engine = create_async_engine(database_url)
    cleanup_factory = async_sessionmaker(cleanup_engine, expire_on_commit=False)
    async with cleanup_factory() as cleanup_session:
        await cleanup_session.execute(delete(MarketCandleRecord))
        await cleanup_session.commit()

    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[raw_kline()])

    settings = Settings.model_validate(
        {
            "DATABASE_URL": database_url,
            "BINANCE_BASE_URL": "https://binance.test",
            "BINANCE_INGESTION_LIMIT": 1,
        }
    )
    result = await run(
        settings=settings,
        clock=lambda: NOW,
        transport=httpx.MockTransport(handler),
    )

    async with cleanup_factory() as verification_session:
        assets = list(
            await verification_session.scalars(
                select(MarketCandleRecord.asset).order_by(MarketCandleRecord.asset)
            )
        )
        await verification_session.execute(delete(MarketCandleRecord))
        await verification_session.commit()
    await cleanup_engine.dispose()

    assert len(requests) == 2
    assert (
        result.ingestion.received,
        result.ingestion.inserted,
        result.ingestion.duplicates,
    ) == (2, 2, 0)
    assert assets == ["BTC/USDT", "ETH/USDT"]


async def test_feature_failure_preserves_candles_and_next_run_recovers() -> None:
    database_url = get_settings().database_url
    engine = create_async_engine(database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as cleanup:
        await cleanup.execute(delete(MarketFeatureRecord))
        await cleanup.execute(delete(MarketCandleRecord))
        await cleanup.commit()

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=raw_klines(20))

    settings = Settings.model_validate(
        {
            "DATABASE_URL": database_url,
            "BINANCE_BASE_URL": "https://binance.test",
            "BINANCE_INGESTION_LIMIT": 20,
        }
    )
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
        target_start_time: datetime | None = None,
        target_end_time: datetime | None = None,
        context_limit: int | None = None,
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
            target_start_time=target_start_time,
            target_end_time=target_end_time,
            context_limit=context_limit,
        )

    with (
        patch(
            "app.features.orchestration.calculate_market_features",
            fail_after_btc,
        ),
        pytest.raises(RuntimeError) as captured,
    ):
        await run(
            settings=settings,
            clock=lambda: NOW,
            transport=httpx.MockTransport(handler),
        )

    assert captured.value is original
    async with factory() as verification:
        assert (
            len((await verification.scalars(select(MarketCandleRecord.id))).all()) == 40
        )
        assert (
            len((await verification.scalars(select(MarketFeatureRecord.id))).all()) == 0
        )

    recovered = await run(
        settings=settings,
        clock=lambda: NOW,
        transport=httpx.MockTransport(handler),
    )

    async with factory() as verification:
        candle_count = len(
            (await verification.scalars(select(MarketCandleRecord.id))).all()
        )
        feature_assets = list(
            await verification.scalars(
                select(MarketFeatureRecord.asset)
                .distinct()
                .order_by(MarketFeatureRecord.asset)
            )
        )
        await verification.execute(delete(MarketFeatureRecord))
        await verification.execute(delete(MarketCandleRecord))
        await verification.commit()
    await engine.dispose()

    assert recovered.ingestion.duplicates == 40
    assert candle_count == 40
    assert feature_assets == ["BTC/USDT", "ETH/USDT"]
