import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from app.core.config import Settings, get_settings
from app.core.database.engine import create_database_engine
from app.core.database.session import create_session_factory
from app.core.logging import configure_logging
from app.paper.cycle_orchestration import CycleResult, run_instrument_cycle
from app.paper.decision_cycles import (
    MVP_INSTRUMENTS,
    Instrument,
    latest_completed_boundary,
)

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


async def run(
    *,
    settings: Settings | None = None,
    clock: Callable[[], datetime] = utc_now,
    instruments: Sequence[Instrument] = MVP_INSTRUMENTS,
) -> tuple[CycleResult, ...]:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)
    engine = create_database_engine(resolved)
    factory = create_session_factory(engine)
    execution_time = clock()
    boundary = latest_completed_boundary(execution_time)
    results: list[CycleResult] = []
    failures: list[Exception] = []
    try:
        for instrument in instruments:
            async with factory() as session:
                try:
                    result = await run_instrument_cycle(
                        session=session,
                        instrument=instrument,
                        boundary=boundary,
                        execution_time=execution_time,
                    )
                    await session.commit()
                    results.append(result)
                    logger.info(
                        "Paper Decision Cycle completed",
                        extra={"structured_fields": _fields(instrument, result)},
                    )
                except Exception as error:
                    try:
                        await session.rollback()
                    except Exception as rollback_error:
                        error.add_note(f"Rollback also failed: {rollback_error!r}")
                    failures.append(error)
                    logger.exception(
                        "Paper Decision Cycle failed",
                        extra={
                            "structured_fields": {
                                "asset": instrument.asset.value,
                                "provider": instrument.provider.value,
                                "market_type": instrument.market_type.value,
                                "interval": instrument.interval.value,
                                "boundary": boundary.isoformat(),
                                "failure_type": type(error).__name__,
                            }
                        },
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


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
