import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from app.core.config import Settings, get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.scores.calculation import ScoreCalculationResult
from app.scores.orchestration import calculate_all_market_scores

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


async def run(
    *, settings: Settings | None = None, clock: Callable[[], datetime] = utc_now
) -> tuple[ScoreCalculationResult, ...]:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    engine = create_database_engine(resolved)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            calculated_at = clock()
            try:
                results = await calculate_all_market_scores(
                    session=session, calculated_at=calculated_at
                )
                await session.commit()
            except Exception as error:
                try:
                    await session.rollback()
                except Exception as rollback_error:
                    error.add_note(f"Rollback also failed: {rollback_error!r}")
                raise
        logger.info(
            "Market score calculation completed",
            extra={
                "structured_fields": {
                    "job_name": "calculate_market_scores",
                    "calculated_at": calculated_at.isoformat(),
                    "scores_calculated": sum(
                        item.scores_calculated for item in results
                    ),
                    "scores_inserted": sum(item.scores_inserted for item in results),
                    "duplicates": sum(item.duplicates for item in results),
                    "model_name": "baseline_technical",
                    "model_version": 1,
                    "assets": [
                        {
                            "asset": item.asset.value,
                            "calculated": item.scores_calculated,
                            "inserted": item.scores_inserted,
                            "duplicates": item.duplicates,
                        }
                        for item in results
                    ],
                }
            },
        )
        return results
    except Exception:
        logger.exception("Market score calculation job failed")
        raise
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
