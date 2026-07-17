import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from app.core.config import Settings, get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.decisions.generation import (
    DecisionGenerationResult,
    PortfolioStateSource,
    PortfolioStateSourceUnavailableError,
    generate_all_market_decisions,
)

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


async def run(
    *,
    settings: Settings | None = None,
    clock: Callable[[], datetime] = utc_now,
    portfolio_state_source: PortfolioStateSource | None = None,
) -> tuple[DecisionGenerationResult, ...]:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    engine = create_database_engine(resolved)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            try:
                if portfolio_state_source is None:
                    raise PortfolioStateSourceUnavailableError(
                        "Decision generation is disabled until a persisted Paper "
                        "Position source is configured"
                    )
                decision_timestamp = clock()
                results = await generate_all_market_decisions(
                    session=session,
                    portfolio_state_source=portfolio_state_source,
                    decision_timestamp=decision_timestamp,
                )
                await session.commit()
            except Exception as error:
                try:
                    await session.rollback()
                except Exception as rollback_error:
                    error.add_note(f"Rollback also failed: {rollback_error!r}")
                raise
        logger.info(
            "Market decision generation completed",
            extra={
                "structured_fields": {
                    "job_name": "generate_market_decisions",
                    "decision_timestamp": decision_timestamp.isoformat(),
                    "calculated": sum(item.calculated for item in results),
                    "inserted": sum(item.inserted for item in results),
                    "duplicates": sum(item.duplicates for item in results),
                }
            },
        )
        return results
    except Exception:
        logger.exception("Market decision generation job failed")
        raise
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
