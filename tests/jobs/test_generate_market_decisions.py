from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.decisions.generation import PortfolioStateSourceUnavailableError
from app.jobs import generate_market_decisions as job

pytestmark = pytest.mark.asyncio


async def test_job_fails_closed_rolls_back_and_cleans_up() -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    session = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job, "create_session_factory", return_value=Mock(return_value=context)
        ),
        pytest.raises(PortfolioStateSourceUnavailableError, match="disabled"),
    ):
        await job.run()

    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once_with()
    engine.dispose.assert_awaited_once_with()
