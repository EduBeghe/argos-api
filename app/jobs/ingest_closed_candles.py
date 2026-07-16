import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings, get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.features.calculation import FeatureCalculationResult
from app.features.orchestration import calculate_all_market_features
from app.ingestion.binance.client import BinanceClient
from app.ingestion.orchestration import (
    ClosedCandleIngestionResult,
    ingest_latest_closed_klines,
)
from app.market.enums import Asset, Interval

JOB_NAME = "ingest_closed_candles"
ASSETS = (Asset.BTC_USDT, Asset.ETH_USDT)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestionPipelineResult:
    ingestion: ClosedCandleIngestionResult
    calculated_at: datetime
    features: tuple[FeatureCalculationResult, ...]


def utc_now() -> datetime:
    """Read the current UTC time at the executable boundary."""

    return datetime.now(UTC)


async def run(
    *,
    settings: Settings | None = None,
    clock: Callable[[], datetime] = utc_now,
    transport: httpx.AsyncBaseTransport | None = None,
    engine: AsyncEngine | None = None,
) -> IngestionPipelineResult:
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
            try:
                ingestion = await ingest_latest_closed_klines(
                    session=session,
                    client=client,
                    assets=ASSETS,
                    interval=Interval.FIVE_MINUTES,
                    now=now,
                    limit=resolved_settings.binance_ingestion_limit,
                )
                await session.commit()
            except Exception as error:
                await _rollback_preserving(session.rollback, error)
                logger.exception("Candle ingestion stage failed")
                raise

            calculated_at = clock()
            try:
                features = await calculate_all_market_features(
                    session=session, calculated_at=calculated_at
                )
                await session.commit()
            except Exception as error:
                await _rollback_preserving(session.rollback, error)
                logger.exception("Feature calculation stage failed")
                raise
        result = IngestionPipelineResult(
            ingestion=ingestion, calculated_at=calculated_at, features=features
        )
        _log_completion(result)
        return result
    finally:
        await database_engine.dispose()


async def _rollback_preserving(
    rollback: Callable[[], Awaitable[None]], original_error: Exception
) -> None:
    try:
        await rollback()
    except Exception as rollback_error:
        original_error.add_note(f"Rollback also failed: {rollback_error!r}")


def _log_completion(result: IngestionPipelineResult) -> None:
    ingestion = result.ingestion
    per_asset = [
        {
            "asset": item.asset.value,
            "received": item.result.received,
            "inserted": item.result.inserted,
            "duplicates": item.result.duplicates,
        }
        for item in ingestion.assets
    ]
    logger.info(
        "Closed candle ingestion job completed",
        extra={
            "structured_fields": {
                "job_name": JOB_NAME,
                "ingestion_completed": True,
                "ingested_at": ingestion.ingested_at.isoformat(),
                "start_time": ingestion.start_time,
                "end_time": ingestion.end_time,
                "received": ingestion.received,
                "inserted": ingestion.inserted,
                "duplicates": ingestion.duplicates,
                "ingestion_assets": per_asset,
                "feature_calculated_at": result.calculated_at.isoformat(),
                "feature_values_calculated": sum(
                    item.values_calculated for item in result.features
                ),
                "feature_values_inserted": sum(
                    item.values_inserted for item in result.features
                ),
                "feature_duplicates": sum(item.duplicates for item in result.features),
                "feature_assets": [
                    {
                        "asset": item.asset.value,
                        "calculated": item.values_calculated,
                        "inserted": item.values_inserted,
                        "duplicates": item.duplicates,
                    }
                    for item in result.features
                ],
            }
        },
    )


def main() -> None:
    """Execute the asynchronous job once."""

    asyncio.run(run())


if __name__ == "__main__":
    main()
