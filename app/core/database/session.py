from collections.abc import AsyncIterator
from typing import cast

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


def create_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create the asynchronous database session factory."""

    return async_sessionmaker(engine, expire_on_commit=False)


def get_session_factory(
    request: Request,
) -> async_sessionmaker[AsyncSession]:
    """Retrieve the session factory initialized during application lifespan."""

    return cast(async_sessionmaker[AsyncSession], request.app.state.session_factory)


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Provide one asynchronous database session per request."""

    session_factory = get_session_factory(request)
    async with session_factory() as session:
        yield session
