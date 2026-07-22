from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.optimization.weekly_research import (
    release_weekly_research_lock,
    try_weekly_research_lock,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def sessions() -> AsyncIterator[tuple[AsyncSession, AsyncSession]]:
    engine = create_async_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as first, factory() as second:
        yield first, second
        await release_weekly_research_lock(first)
        await release_weekly_research_lock(second)
    await engine.dispose()


async def test_weekly_advisory_lock_prevents_concurrent_jobs_and_releases(
    sessions: tuple[AsyncSession, AsyncSession],
) -> None:
    first, second = sessions
    assert await try_weekly_research_lock(first) is True
    assert await try_weekly_research_lock(second) is False
    await release_weekly_research_lock(first)
    assert await try_weekly_research_lock(second) is True
