import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings, get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.ingestion.binance.client import BinanceClient
from app.ingestion.orchestration import (
    ClosedCandleIngestionResult,
    ingest_latest_closed_klines,
)
from app.market.enums import Asset, Interval

JOB_NAME = "ingest_closed_candles"
ASSETS = (Asset.BTC_USDT, Asset.ETH_USDT)

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Read the current UTC time at the executable boundary."""

    return datetime.now(UTC)


async def run(
    *,
    settings: Settings | None = None,
    clock: Callable[[], datetime] = utc_now,
    transport: httpx.AsyncBaseTransport | None = None,
    engine: AsyncEngine | None = None,
) -> ClosedCandleIngestionResult:
    """Run closed-candle ingestion once and release all owned resources."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    database_engine = engine or create_database_engine(resolved_settings)
    session_factory = create_session_factory(database_engine)

    try:
        async with (
            session_factory() as session,
            BinanceClient(resolved_settings, transport=transport) as client,
        ):
            now = clock()
            result = await ingest_latest_closed_klines(
                session=session,
                client=client,
                assets=ASSETS,
                interval=Interval.FIVE_MINUTES,
                now=now,
                limit=resolved_settings.binance_ingestion_limit,
            )
        _log_completion(result)
        return result
    except Exception:
        logger.exception(
            "Closed candle ingestion job failed",
            extra={"structured_fields": {"job_name": JOB_NAME}},
        )
        raise
    finally:
        await database_engine.dispose()


def _log_completion(result: ClosedCandleIngestionResult) -> None:
    per_asset = [
        {
            "asset": item.asset.value,
            "received": item.result.received,
            "inserted": item.result.inserted,
            "duplicates": item.result.duplicates,
        }
        for item in result.assets
    ]
    logger.info(
        "Closed candle ingestion job completed",
        extra={
            "structured_fields": {
                "job_name": JOB_NAME,
                "ingested_at": result.ingested_at.isoformat(),
                "start_time": result.start_time,
                "end_time": result.end_time,
                "received": result.received,
                "inserted": result.inserted,
                "duplicates": result.duplicates,
                "assets": per_asset,
            }
        },
    )


def main() -> None:
    """Execute the asynchronous job once."""

    asyncio.run(run())


if __name__ == "__main__":
    main()
