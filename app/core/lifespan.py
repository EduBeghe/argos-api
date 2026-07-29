import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.auth.config import get_auth_settings
from app.core.config import get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and dispose shared application resources."""

    settings = get_settings()
    auth_settings = get_auth_settings()
    configure_logging(settings.log_level)
    auth_settings.validate_startup()
    if not auth_settings.enforcement_enabled:
        logging.getLogger(__name__).warning(
            "Authentication enforcement is disabled; API routes remain public"
        )
    engine = create_database_engine(settings)
    app.state.session_factory = create_session_factory(engine)

    try:
        yield
    finally:
        await engine.dispose()
