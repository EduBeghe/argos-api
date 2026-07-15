from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings


def create_database_engine(settings: Settings) -> AsyncEngine:
    """Create the application's asynchronous SQLAlchemy engine."""

    return create_async_engine(settings.database_url, pool_pre_ping=True)
