from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from app.jobs import generate_market_decisions as job
from app.market.enums import Asset, Interval, MarketType, Provider
from app.paper.cycle_orchestration import CycleResult
from app.paper.decision_cycles import Instrument

pytestmark = pytest.mark.asyncio
NOW = datetime(2026, 7, 17, 12, 32, tzinfo=UTC)
BTC = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)
ETH = Instrument(
    Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def result(asset: Asset, *, skipped: bool = False) -> CycleResult:
    return CycleResult(
        uuid4(),
        uuid4(),
        datetime(2026, 7, 17, 12, 30, tzinfo=UTC),
        uuid4(),
        uuid4(),
        None,
        "no_action",
        skipped,
    )


def context(session: AsyncMock) -> MagicMock:
    value = MagicMock()
    value.__aenter__ = AsyncMock(return_value=session)
    value.__aexit__ = AsyncMock(return_value=None)
    return value


async def test_job_commits_each_instrument_logs_and_cleans_up() -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    btc_session = AsyncMock()
    eth_session = AsyncMock()
    factory = Mock(side_effect=[context(btc_session), context(eth_session)])
    run_cycle = AsyncMock(
        side_effect=[result(Asset.BTC_USDT), result(Asset.ETH_USDT, skipped=True)]
    )
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(job, "create_session_factory", return_value=factory),
        patch.object(job, "run_instrument_cycle", run_cycle),
        patch.object(job.logger, "info") as logged,
    ):
        results = await job.run(clock=lambda: NOW, instruments=(BTC, ETH))
    assert len(results) == 2
    btc_session.commit.assert_awaited_once_with()
    eth_session.commit.assert_awaited_once_with()
    btc_session.rollback.assert_not_awaited()
    eth_session.rollback.assert_not_awaited()
    assert logged.call_count == 2
    engine.dispose.assert_awaited_once_with()


async def test_instrument_failure_rolls_back_without_undoing_success() -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    btc_session = AsyncMock()
    eth_session = AsyncMock()
    failure = RuntimeError("BTC evidence unavailable")
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job,
            "create_session_factory",
            return_value=Mock(side_effect=[context(btc_session), context(eth_session)]),
        ),
        patch.object(
            job,
            "run_instrument_cycle",
            AsyncMock(side_effect=[failure, result(Asset.ETH_USDT)]),
        ),
        patch.object(job.logger, "exception"),
        pytest.raises(ExceptionGroup),
    ):
        await job.run(clock=lambda: NOW, instruments=(BTC, ETH))
    btc_session.rollback.assert_awaited_once_with()
    btc_session.commit.assert_not_awaited()
    eth_session.commit.assert_awaited_once_with()
    eth_session.rollback.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()
