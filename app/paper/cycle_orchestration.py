from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.decisions.generation import generate_decision_from_evidence
from app.market.catalog import Instrument
from app.paper.calculation import PaperExecutionError
from app.paper.config import DEFAULT_ACCOUNT_CURRENCY
from app.paper.decision_cycles import (
    DecisionCycleError,
    DecisionCycleStatus,
    PaperDecisionCycleRecord,
    claim_cycle,
    get_boundary_score_evidence,
    get_cycle_by_identity,
    validate_cycle_start,
)
from app.paper.decision_evidence import persist_baseline_decision_identity
from app.paper.deployment_models import PaperDeploymentRecord
from app.paper.execution import execute_decision
from app.paper.models import NoTrade
from app.paper.persistence import PaperTradeRecord, get_active_run, get_run
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
    source_score_id: UUID | None
    decision_snapshot_id: UUID
    decision_identity_id: UUID
    decision_kind: str
    trade_id: UUID | None
    action: str
    skipped: bool


async def run_instrument_cycle(
    *,
    session: AsyncSession,
    instrument: Instrument,
    boundary: datetime,
    execution_time: datetime,
    run_id: UUID | None = None,
    deployment_id: UUID | None = None,
) -> CycleResult:
    validate_cycle_start(boundary=boundary, execution_time=execution_time)
    if deployment_id is not None:
        deployment = await session.scalar(
            select(PaperDeploymentRecord)
            .where(PaperDeploymentRecord.id == deployment_id)
            .with_for_update()
        )
        if deployment is None or deployment.lifecycle != "active":
            raise DecisionCycleError("baseline deployment is not active")
        if (
            deployment.current_run_id != run_id
            or deployment.strategy_kind != "baseline"
        ):
            raise DecisionCycleError("baseline deployment lineage is incompatible")
    run = (
        await get_run(session, run_id=run_id)
        if run_id is not None
        else await get_active_run(session, currency=DEFAULT_ACCOUNT_CURRENCY, lock=True)
    )
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
    portfolio = PersistedPaperPortfolioStateSource(session, run.id)
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
    identity = await persist_baseline_decision_identity(session, generated.snapshot.id)
    execution = await execute_decision(
        session=session,
        decision=generated.snapshot,
        executed_at=execution_time,
        run_id=run.id,
        decision_identity_id=identity.id,
    )
    trade_id = execution.id if isinstance(execution, PaperTradeRecord) else None
    if not isinstance(execution, (PaperTradeRecord, NoTrade)):
        raise DecisionCycleError("unexpected Paper execution result")
    cycle.source_score_id = evidence.score.id
    cycle.decision_snapshot_id = generated.snapshot.id
    cycle.ml_decision_snapshot_id = None
    cycle.decision_identity_id = identity.id
    cycle.decision_kind = "baseline"
    cycle.paper_trade_id = trade_id
    cycle.result = generated.snapshot.action.value
    cycle.completed_at = execution_time
    cycle.status = DecisionCycleStatus.COMPLETED.value
    await session.flush()
    return _result(cycle, skipped=False)


def _result(cycle: PaperDecisionCycleRecord, *, skipped: bool) -> CycleResult:
    if (
        cycle.decision_identity_id is None
        or cycle.decision_kind is None
        or cycle.result is None
    ):
        raise DecisionCycleError("completed Decision Cycle is incomplete")
    return CycleResult(
        cycle_id=cycle.id,
        run_id=cycle.paper_run_id,
        boundary=cycle.scheduled_boundary,
        source_score_id=cycle.source_score_id,
        decision_snapshot_id=(
            cycle.decision_snapshot_id
            if cycle.decision_snapshot_id is not None
            else cycle.ml_decision_snapshot_id
        )
        or (_raise_incomplete()),
        decision_identity_id=cycle.decision_identity_id,
        decision_kind=cycle.decision_kind,
        trade_id=cycle.paper_trade_id,
        action=cycle.result,
        skipped=skipped,
    )


def _raise_incomplete() -> UUID:
    raise DecisionCycleError("completed Decision Cycle has no typed snapshot")
