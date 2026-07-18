from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.generation import generate_decision_from_evidence
from app.paper.calculation import PaperExecutionError
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.paper.decision_cycles import (
    DecisionCycleError,
    DecisionCycleStatus,
    Instrument,
    PaperDecisionCycleRecord,
    claim_cycle,
    get_boundary_score_evidence,
    get_cycle_by_identity,
    validate_cycle_start,
)
from app.paper.execution import execute_decision
from app.paper.models import NoTrade
from app.paper.persistence import PaperTradeRecord, get_active_run
from app.paper.portfolio import PersistedPaperPortfolioStateSource
from app.strategies.baseline_long_only import (
    CONFIGURATION_VERSION,
    STRATEGY_NAME,
    STRATEGY_VERSION,
)


@dataclass(frozen=True, slots=True)
class CycleResult:
    cycle_id: UUID
    run_id: UUID
    boundary: datetime
    source_score_id: UUID
    decision_snapshot_id: UUID
    trade_id: UUID | None
    action: str
    skipped: bool


async def run_instrument_cycle(
    *,
    session: AsyncSession,
    instrument: Instrument,
    boundary: datetime,
    execution_time: datetime,
) -> CycleResult:
    validate_cycle_start(boundary=boundary, execution_time=execution_time)
    run = await get_active_run(session, currency=DEFAULT_ACCOUNT_CURRENCY, lock=True)
    if run is None:
        raise PaperExecutionError("active Paper Trading run not found")
    if boundary < run.started_at:
        raise DecisionCycleError("scheduled boundary predates the active run")
    cycle = await claim_cycle(
        session=session,
        run_id=run.id,
        instrument=instrument,
        boundary=boundary,
        started_at=execution_time,
        strategy_name=STRATEGY_NAME,
        strategy_version=STRATEGY_VERSION,
        configuration_version=CONFIGURATION_VERSION,
    )
    if cycle is None:
        existing = await get_cycle_by_identity(
            session=session,
            run_id=run.id,
            instrument=instrument,
            boundary=boundary,
            strategy_name=STRATEGY_NAME,
            strategy_version=STRATEGY_VERSION,
            configuration_version=CONFIGURATION_VERSION,
        )
        if existing is None or existing.status != DecisionCycleStatus.COMPLETED.value:
            raise DecisionCycleError("Decision Cycle is already being processed")
        return _result(existing, skipped=True)

    evidence = await get_boundary_score_evidence(
        session=session,
        instrument=instrument,
        boundary=boundary,
        run_started_at=run.started_at,
    )
    portfolio = PersistedPaperPortfolioStateSource(session)
    state = await portfolio.get_portfolio_state(
        asset=instrument.asset,
        provider=instrument.provider,
        market_type=instrument.market_type,
        interval=instrument.interval,
    )
    generated = await generate_decision_from_evidence(
        session=session,
        evidence=evidence,
        portfolio_state=state,
        decision_timestamp=boundary,
    )
    execution = await execute_decision(
        session=session,
        decision=generated.snapshot,
        executed_at=execution_time,
    )
    trade_id = execution.id if isinstance(execution, PaperTradeRecord) else None
    if not isinstance(execution, (PaperTradeRecord, NoTrade)):
        raise DecisionCycleError("unexpected Paper execution result")
    cycle.source_score_id = evidence.score.id
    cycle.decision_snapshot_id = generated.snapshot.id
    cycle.paper_trade_id = trade_id
    cycle.result = generated.snapshot.action.value
    cycle.completed_at = execution_time
    cycle.status = DecisionCycleStatus.COMPLETED.value
    await session.flush()
    return _result(cycle, skipped=False)


def _result(cycle: PaperDecisionCycleRecord, *, skipped: bool) -> CycleResult:
    if (
        cycle.source_score_id is None
        or cycle.decision_snapshot_id is None
        or cycle.result is None
    ):
        raise DecisionCycleError("completed Decision Cycle is incomplete")
    return CycleResult(
        cycle_id=cycle.id,
        run_id=cycle.paper_run_id,
        boundary=cycle.scheduled_boundary,
        source_score_id=cycle.source_score_id,
        decision_snapshot_id=cycle.decision_snapshot_id,
        trade_id=cycle.paper_trade_id,
        action=cycle.result,
        skipped=skipped,
    )

