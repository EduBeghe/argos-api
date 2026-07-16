from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.jobs.ingest_closed_candles import run
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
    assert (result.received, result.inserted, result.duplicates) == (2, 2, 0)
    assert assets == ["BTC/USDT", "ETH/USDT"]
