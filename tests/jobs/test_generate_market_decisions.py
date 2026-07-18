from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from app.jobs import generate_market_decisions as job
from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.paper.cycle_orchestration import CycleResult
from app.paper.equity_snapshots import (
    PaperPortfolioSnapshotRecord,
    SnapshotWriteResult,
)

pytestmark = pytest.mark.asyncio
NOW = datetime(2026, 7, 17, 12, 32, tzinfo=UTC)
BTC = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)
ETH = Instrument(
    Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def result(
    asset: Asset, *, skipped: bool = False, run_id: UUID | None = None
) -> CycleResult:
    return CycleResult(
        uuid4(),
        run_id or uuid4(),
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
    snapshot_session = AsyncMock()
    factory = Mock(
        side_effect=[
            context(btc_session),
            context(eth_session),
            context(snapshot_session),
        ]
    )
    shared_run_id = uuid4()
    btc_result = result(Asset.BTC_USDT, run_id=shared_run_id)
    eth_result = result(Asset.ETH_USDT, skipped=True, run_id=shared_run_id)
    run_cycle = AsyncMock(side_effect=[btc_result, eth_result])
    with (
        patch.object(job, "configure_logging"),
        patch.object(
            job, "enabled_decision_instruments", return_value=(BTC, ETH)
        ) as projected,
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(job, "create_session_factory", return_value=factory),
        patch.object(job, "run_instrument_cycle", run_cycle),
        patch.object(job, "get_run", AsyncMock(return_value=SimpleNamespace())),
        patch.object(
            job,
            "persist_equity_snapshot",
            AsyncMock(
                return_value=SnapshotWriteResult(
                    PaperPortfolioSnapshotRecord(
                        run_id=btc_result.run_id,
                        id=uuid4(),
                        scheduled_boundary=datetime(2026, 7, 17, 12, 30, tzinfo=UTC),
                        recorded_at=NOW,
                        account_currency="USDT",
                        initial_cash=10000,
                        current_cash=10000,
                        positions_market_value=0,
                        positions=[],
                        equity=10000,
                        total_pnl=0,
                        total_return=0,
                        configuration_version=1,
                    ),
                    True,
                )
            ),
        ),
        patch.object(job.logger, "info") as logged,
    ):
        results = await job.run(clock=lambda: NOW)
    assert len(results) == 2
    btc_session.commit.assert_awaited_once_with()
    eth_session.commit.assert_awaited_once_with()
    btc_session.rollback.assert_not_awaited()
    eth_session.rollback.assert_not_awaited()
    snapshot_session.commit.assert_awaited_once_with()
    assert logged.call_count == 3
    projected.assert_called_once_with()
    engine.dispose.assert_awaited_once_with()


async def test_snapshot_failure_does_not_rollback_committed_cycles() -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    btc_session = AsyncMock()
    eth_session = AsyncMock()
    snapshot_session = AsyncMock()
    snapshot_error = RuntimeError("snapshot failed")
    shared_run_id = uuid4()
    with (
        patch.object(job, "configure_logging"),
        patch.object(job, "create_database_engine", return_value=engine),
        patch.object(
            job,
            "create_session_factory",
            return_value=Mock(
                side_effect=[
                    context(btc_session),
                    context(eth_session),
                    context(snapshot_session),
                ]
            ),
        ),
        patch.object(
            job,
            "run_instrument_cycle",
            AsyncMock(
                side_effect=[
                    result(Asset.BTC_USDT, run_id=shared_run_id),
                    result(Asset.ETH_USDT, run_id=shared_run_id),
                ]
            ),
        ),
        patch.object(job, "get_run", AsyncMock(return_value=SimpleNamespace())),
        patch.object(
            job, "persist_equity_snapshot", AsyncMock(side_effect=snapshot_error)
        ),
        patch.object(job.logger, "info"),
        pytest.raises(RuntimeError, match="snapshot failed"),
    ):
        await job.run(clock=lambda: NOW, instruments=(BTC, ETH))
    btc_session.commit.assert_awaited_once_with()
    eth_session.commit.assert_awaited_once_with()
    snapshot_session.rollback.assert_awaited_once_with()


async def test_snapshot_log_fields_distinguish_created_and_existing() -> None:
    snapshot = PaperPortfolioSnapshotRecord(
        id=uuid4(),
        run_id=uuid4(),
        scheduled_boundary=NOW,
        recorded_at=NOW,
        account_currency="USDT",
        initial_cash=10000,
        current_cash=10000,
        positions_market_value=0,
        equity=10000,
        total_pnl=0,
        total_return=0,
        configuration_version=1,
        positions=[],
    )
    assert (
        job._snapshot_fields(SnapshotWriteResult(snapshot, True))["status"] == "created"
    )
    assert (
        job._snapshot_fields(SnapshotWriteResult(snapshot, False))["status"]
        == "existing"
    )


async def test_instrument_failure_rolls_back_without_undoing_success() -> None:
    engine = MagicMock()
    engine.dispose = AsyncMock()
    btc_session = AsyncMock()
    eth_session = AsyncMock()
    failure = RuntimeError("BTC evidence unavailable")
    persist_snapshot = AsyncMock()
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
        patch.object(job, "persist_equity_snapshot", persist_snapshot),
        pytest.raises(ExceptionGroup),
    ):
        await job.run(clock=lambda: NOW, instruments=(BTC, ETH))
    btc_session.rollback.assert_awaited_once_with()
    btc_session.commit.assert_not_awaited()
    eth_session.commit.assert_awaited_once_with()
    eth_session.rollback.assert_not_awaited()
    persist_snapshot.assert_not_awaited()
    engine.dispose.assert_awaited_once_with()
