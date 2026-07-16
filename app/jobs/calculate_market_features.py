import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from app.core.config import Settings, get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.features.calculation import FeatureCalculationResult
from app.features.orchestration import calculate_all_market_features

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


async def run(
    *, settings: Settings | None = None, clock: Callable[[], datetime] = utc_now
) -> tuple[FeatureCalculationResult, ...]:
    """Calculate EMA 20 for both MVP assets in one transaction."""

    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    engine = create_database_engine(resolved)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            calculated_at = clock()
            try:
                results = await calculate_all_market_features(
                    session=session, calculated_at=calculated_at
                )
                await session.commit()
            except Exception as original_error:
                try:
                    await session.rollback()
                except Exception as rollback_error:
                    original_error.add_note(f"Rollback also failed: {rollback_error!r}")
                raise
        logger.info(
            "Market feature calculation completed",
            extra={
                "structured_fields": {
                    "job_name": "calculate_market_features",
                    "calculated_at": calculated_at.isoformat(),
                    "values_calculated": sum(
                        item.values_calculated for item in results
                    ),
                    "values_inserted": sum(item.values_inserted for item in results),
                    "duplicates": sum(item.duplicates for item in results),
                    "assets": [
                        {
                            "asset": item.asset.value,
                            "calculated": item.values_calculated,
                            "inserted": item.values_inserted,
                            "duplicates": item.duplicates,
                        }
                        for item in results
                    ],
                }
            },
        )
        return results
    except Exception:
        logger.exception("Market feature calculation job failed")
        raise
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
