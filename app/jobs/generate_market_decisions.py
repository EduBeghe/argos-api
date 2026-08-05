import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.market.catalog import Instrument, enabled_decision_instruments
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.protected.models import MLStrategySpecificationRecord
from app.paper.cycle_orchestration import CycleResult, run_instrument_cycle
from app.paper.decision_cycles import latest_completed_boundary
from app.paper.deployment_models import PaperDeploymentRecord
from app.paper.equity_snapshots import SnapshotWriteResult, persist_equity_snapshot
from app.paper.ml_cycle import run_ml_instrument_cycle
from app.paper.persistence import get_run

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


async def run(
    *,
    settings: Settings | None = None,
    clock: Callable[[], datetime] = utc_now,
    instruments: Sequence[Instrument] | None = None,
) -> tuple[CycleResult, ...]:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    engine = create_database_engine(resolved)
    factory = create_session_factory(engine)
    execution_time = clock()
    boundary = latest_completed_boundary(execution_time)
    results: list[CycleResult] = []
    failures: list[Exception] = []
    selected_instruments = (
        enabled_decision_instruments() if instruments is None else instruments
    )
    try:
        deployments = await _load_active_deployments(factory)
        for deployment in deployments:
            deployment_results: list[CycleResult] = []
            deployment_failed = False
            deployment_instruments = await _deployment_instruments(
                factory, deployment, selected_instruments
            )
            for instrument in deployment_instruments:
                async with factory() as session:
                    try:
                        if deployment.strategy_kind == "baseline":
                            result = await run_instrument_cycle(
                                session=session,
                                instrument=instrument,
                                boundary=boundary,
                                execution_time=execution_time,
                                run_id=deployment.current_run_id,
                                deployment_id=deployment.id,
                            )
                        else:
                            result = await run_ml_instrument_cycle(
                                session=session,
                                deployment_id=deployment.id,
                                instrument=instrument,
                                boundary=boundary,
                                execution_time=execution_time,
                            )
                        await session.commit()
                        results.append(result)
                        deployment_results.append(result)
                        logger.info(
                            "Paper Decision Cycle completed",
                            extra={
                                "structured_fields": {
                                    **_fields(instrument, result),
                                    "deployment_id": str(deployment.id),
                                }
                            },
                        )
                    except Exception as error:
                        try:
                            await session.rollback()
                        except Exception as rollback_error:
                            error.add_note(f"Rollback also failed: {rollback_error!r}")
                        failures.append(error)
                        deployment_failed = True
                        logger.exception(
                            "Paper Decision Cycle failed",
                            extra={
                                "structured_fields": {
                                    "deployment_id": str(deployment.id),
                                    "asset": instrument.asset.value,
                                    "provider": instrument.provider.value,
                                    "market_type": instrument.market_type.value,
                                    "interval": instrument.interval.value,
                                    "boundary": boundary.isoformat(),
                                    "failure_type": type(error).__name__,
                                }
                            },
                        )
            if deployment_results and not deployment_failed:
                await _persist_snapshot(
                    factory=factory,
                    run_id=deployment_results[0].run_id,
                    boundary=boundary,
                    recorded_at=clock(),
                )
        if failures:
            raise ExceptionGroup("one or more Paper Decision Cycles failed", failures)
        return tuple(results)
    finally:
        await engine.dispose()


def _fields(instrument: Instrument, result: CycleResult) -> dict[str, object]:
    return {
        "asset": instrument.asset.value,
        "provider": instrument.provider.value,
        "market_type": instrument.market_type.value,
        "interval": instrument.interval.value,
        "run_id": str(result.run_id),
        "boundary": result.boundary.isoformat(),
        "source_score_id": str(result.source_score_id),
        "action": result.action,
        "decision_snapshot_id": str(result.decision_snapshot_id),
        "trade_id": str(result.trade_id) if result.trade_id else None,
        "status": "skipped" if result.skipped else "completed",
    }


async def _persist_snapshot(
    *,
    factory: async_sessionmaker[AsyncSession],
    run_id: UUID,
    boundary: datetime,
    recorded_at: datetime,
) -> None:
    async with factory() as session:
        try:
            run = await get_run(session, run_id=run_id)
            if run is None:
                raise RuntimeError("Paper Trading run not found for equity snapshot")
            result = await persist_equity_snapshot(
                session=session,
                run=run,
                scheduled_boundary=boundary,
                recorded_at=recorded_at,
            )
            await session.commit()
        except Exception as error:
            try:
                await session.rollback()
            except Exception as rollback_error:
                error.add_note(f"Snapshot rollback also failed: {rollback_error!r}")
            raise
    logger.info(
        "Paper portfolio snapshot persisted",
        extra={"structured_fields": _snapshot_fields(result)},
    )


async def _load_active_deployments(
    factory: async_sessionmaker[AsyncSession],
) -> tuple[PaperDeploymentRecord, ...]:
    async with factory() as session:
        return tuple(
            (
                await session.scalars(
                    select(PaperDeploymentRecord)
                    .where(PaperDeploymentRecord.lifecycle == "active")
                    .order_by(PaperDeploymentRecord.id)
                )
            ).all()
        )


async def _deployment_instruments(
    factory: async_sessionmaker[AsyncSession],
    deployment: PaperDeploymentRecord,
    baseline_instruments: Sequence[Instrument],
) -> tuple[Instrument, ...]:
    if deployment.strategy_kind == "baseline":
        return tuple(baseline_instruments)
    if deployment.strategy_specification_id is None:
        raise RuntimeError("ML deployment has no Strategy specification")
    async with factory() as session:
        specification = await session.get(
            MLStrategySpecificationRecord, deployment.strategy_specification_id
        )
    identity = specification.specification.get("instrument") if specification else None
    if not isinstance(identity, dict):
        raise RuntimeError("ML Strategy specification has no instrument identity")
    try:
        return (
            Instrument(
                Asset(str(identity["asset"])),
                Provider(str(identity["provider"])),
                MarketType(str(identity["market_type"])),
                Interval(str(identity["interval"])),
            ),
        )
    except (KeyError, ValueError) as error:
        raise RuntimeError("ML Strategy instrument identity is invalid") from error


def _snapshot_fields(result: SnapshotWriteResult) -> dict[str, object]:
    snapshot = result.snapshot
    return {
        "run_id": str(snapshot.run_id),
        "snapshot_id": str(snapshot.id),
        "scheduled_boundary": snapshot.scheduled_boundary.isoformat(),
        "recorded_at": snapshot.recorded_at.isoformat(),
        "positions_count": len(snapshot.positions),
        "equity": str(snapshot.equity),
        "total_pnl": str(snapshot.total_pnl),
        "status": "created" if result.created else "existing",
    }


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
