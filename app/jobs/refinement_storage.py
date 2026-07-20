import asyncio
import logging

from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.optimization.memory import refinement_storage_diagnostics

logger = logging.getLogger(__name__)


async def run() -> dict[str, int]:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_database_engine(settings)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            values = await refinement_storage_diagnostics(session)
    finally:
        await engine.dispose()
    logger.info(
        "Strategy refinement storage diagnostics",
        extra={"structured_fields": values},
    )
    return values


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
