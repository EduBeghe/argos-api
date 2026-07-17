import argparse
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from app.jobs import reset_paper_account as job


def test_decimal_argument_rejects_malformed_input() -> None:
    assert job.decimal_argument("10000.25") == Decimal("10000.25")
    with pytest.raises(argparse.ArgumentTypeError):
        job.decimal_argument("not-money")


@pytest.mark.asyncio
async def test_job_commits_logs_uses_clock_and_cleans_up() -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    session = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    now = datetime(2026, 7, 17, tzinfo=UTC)
    reset = AsyncMock()
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job, "create_session_factory", return_value=Mock(return_value=context)
        ),
        patch.object(job, "reset_paper_account", reset),
        patch.object(job.logger, "info") as logged,
    ):
        await job.run(
            initial_cash=Decimal("12000"), reason="baseline", clock=lambda: now
        )
    reset.assert_awaited_once_with(
        session=session,
        started_at=now,
        initial_cash=Decimal("12000"),
        reason="baseline",
    )
    session.commit.assert_awaited_once_with()
    session.rollback.assert_not_awaited()
    logged.assert_called_once()
    engine.dispose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_job_rolls_back_logs_and_cleans_up_on_failure() -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    session = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    failure = RuntimeError("database unavailable")
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job, "create_session_factory", return_value=Mock(return_value=context)
        ),
        patch.object(job, "reset_paper_account", AsyncMock(side_effect=failure)),
        patch.object(job.logger, "exception") as logged,
        pytest.raises(RuntimeError, match="database unavailable"),
    ):
        await job.run(initial_cash=Decimal("10000"), reason=None)
    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once_with()
    logged.assert_called_once()
    engine.dispose.assert_awaited_once_with()
