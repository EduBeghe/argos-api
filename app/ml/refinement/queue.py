from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.contract import MISSING_DATA_POLICY_VERSION, VARIABLE_REGISTRY_VERSION
from app.ml.screening.engines import library_versions

from .contracts import (
    ML_REFINEMENT_CLASSIFICATION_VERSION,
    ML_REFINEMENT_CONTRACT_VERSION,
    MLRefinementRequest,
)
from .models import MLRefinementRunRecord
from .persistence import completed_run_is_complete, experiment_key
from .planning import build_refinement_plan


class QueueDisposition(StrEnum):
    QUEUED = "queued"
    ALREADY_QUEUED = "already_queued"
    ALREADY_RUNNING = "already_running"
    REUSED_COMPLETED = "reused_completed"
    REQUEUED_FAILED = "requeued_failed"
    REQUEUED_CANCELLED = "requeued_cancelled"


class RefinementQueueNotFoundError(LookupError):
    pass


class RefinementQueueConflictError(RuntimeError):
    pass


class RefinementQueueValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    run: MLRefinementRunRecord
    disposition: QueueDisposition


def _reset_values(*, selected: int, omitted: int) -> dict[str, object]:
    return {
        "status": "queued",
        "evaluations_selected": selected,
        "evaluations_computed": 0,
        "evaluations_reused": 0,
        "evaluations_omitted": omitted,
        "budget_exhausted": omitted > 0,
        "termination_reason": None,
        "started_at": None,
        "completed_at": None,
        "failed_at": None,
        "failure_type": None,
        "failure_message": None,
        "elapsed_seconds": None,
        "worker_id": None,
        "claimed_at": None,
        "heartbeat_at": None,
        "cancellation_requested_at": None,
    }


async def enqueue_refinement(
    session: AsyncSession,
    request: MLRefinementRequest,
    *,
    now: datetime | None = None,
) -> EnqueueResult:
    from .persistence import resolve_source

    source, _ = await resolve_source(session, request)
    try:
        plan = build_refinement_plan(request)
    except ValueError as error:
        raise RefinementQueueValidationError(str(error)) from error
    if not plan.evaluations:
        raise RefinementQueueValidationError("refinement plan selected no evaluations")
    versions = library_versions()
    key = experiment_key(
        request=request,
        dataset_fingerprint=source.dataset_fingerprint,
        versions=versions,
    )
    existing = await session.scalar(
        select(MLRefinementRunRecord).where(MLRefinementRunRecord.experiment_key == key)
    )
    if existing is not None:
        if existing.status == "queued":
            return EnqueueResult(existing, QueueDisposition.ALREADY_QUEUED)
        if existing.status == "running":
            return EnqueueResult(existing, QueueDisposition.ALREADY_RUNNING)
        if existing.status == "completed" and await completed_run_is_complete(
            session, existing
        ):
            return EnqueueResult(existing, QueueDisposition.REUSED_COMPLETED)
        if existing.status not in {"failed", "cancelled"}:
            raise RefinementQueueConflictError("identical completed run is incomplete")
        disposition = (
            QueueDisposition.REQUEUED_FAILED
            if existing.status == "failed"
            else QueueDisposition.REQUEUED_CANCELLED
        )
        await session.execute(
            update(MLRefinementRunRecord)
            .where(MLRefinementRunRecord.id == existing.id)
            .values(
                **_reset_values(
                    selected=len(plan.evaluations), omitted=plan.omitted_by_trial_budget
                )
            )
        )
        await session.commit()
        await session.refresh(existing)
        return EnqueueResult(existing, disposition)
    timestamp = now or datetime.now(UTC)
    record = MLRefinementRunRecord(
        id=uuid4(),
        experiment_key=key,
        source_screening_run_id=request.source_screening_run_id,
        source_screening_result_id=request.source_screening_result_id,
        status="queued",
        contract_version=ML_REFINEMENT_CONTRACT_VERSION,
        classification_version=ML_REFINEMENT_CLASSIFICATION_VERSION,
        dataset_fingerprint=source.dataset_fingerprint,
        variable_registry_version=VARIABLE_REGISTRY_VERSION,
        missing_data_policy_version=MISSING_DATA_POLICY_VERSION,
        library_versions=versions,
        normalized_configuration=request.normalized_payload(),
        notes=request.notes,
        maximum_trials=request.maximum_trials,
        maximum_runtime_seconds=request.maximum_runtime_seconds,
        combinations_planned=plan.total_combinations,
        evaluations_selected=len(plan.evaluations),
        evaluations_computed=0,
        evaluations_reused=0,
        evaluations_omitted=plan.omitted_by_trial_budget,
        budget_exhausted=plan.omitted_by_trial_budget > 0,
        termination_reason=None,
        started_at=None,
        completed_at=None,
        failed_at=None,
        failure_type=None,
        failure_message=None,
        worker_id=None,
        claimed_at=None,
        heartbeat_at=None,
        cancellation_requested_at=None,
        elapsed_seconds=None,
        created_at=timestamp,
    )
    session.add(record)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        winner = await session.scalar(
            select(MLRefinementRunRecord).where(
                MLRefinementRunRecord.experiment_key == key
            )
        )
        if winner is None:
            raise
        winner_disposition = {
            "queued": QueueDisposition.ALREADY_QUEUED,
            "running": QueueDisposition.ALREADY_RUNNING,
            "completed": QueueDisposition.REUSED_COMPLETED,
        }.get(winner.status)
        if winner_disposition is None:
            return await enqueue_refinement(session, request, now=timestamp)
        return EnqueueResult(winner, winner_disposition)
    return EnqueueResult(record, QueueDisposition.QUEUED)


async def cancel_refinement(
    session: AsyncSession, run_id: UUID, *, now: datetime | None = None
) -> MLRefinementRunRecord:
    record = await session.get(MLRefinementRunRecord, run_id)
    if record is None:
        raise RefinementQueueNotFoundError(f"ML refinement run {run_id} not found")
    timestamp = now or datetime.now(UTC)
    if record.status == "queued":
        record.status = "cancelled"
        record.completed_at = timestamp
        record.termination_reason = "cancelled_by_user"
    elif record.status == "running" and record.cancellation_requested_at is None:
        record.cancellation_requested_at = timestamp
    await session.commit()
    await session.refresh(record)
    return record


async def retry_refinement(session: AsyncSession, run_id: UUID) -> EnqueueResult:
    record = await session.get(MLRefinementRunRecord, run_id)
    if record is None:
        raise RefinementQueueNotFoundError(f"ML refinement run {run_id} not found")
    if record.status not in {"failed", "cancelled"}:
        raise RefinementQueueConflictError(
            f"{record.status} refinement run cannot be retried"
        )
    disposition = (
        QueueDisposition.REQUEUED_FAILED
        if record.status == "failed"
        else QueueDisposition.REQUEUED_CANCELLED
    )
    selected = record.evaluations_selected
    omitted = max(record.combinations_planned - selected, 0)
    for name, value in _reset_values(selected=selected, omitted=omitted).items():
        setattr(record, name, value)
    await session.commit()
    await session.refresh(record)
    return EnqueueResult(record, disposition)


async def fail_stale_runs(
    session: AsyncSession,
    *,
    stale_after: timedelta,
    now: datetime | None = None,
) -> int:
    timestamp = now or datetime.now(UTC)
    result = await session.execute(
        update(MLRefinementRunRecord)
        .where(
            MLRefinementRunRecord.status == "running",
            MLRefinementRunRecord.heartbeat_at < timestamp - stale_after,
        )
        .values(
            status="failed",
            failed_at=timestamp,
            failure_type="WorkerLeaseExpired",
            failure_message=(
                f"worker heartbeat exceeded {int(stale_after.total_seconds())} seconds"
            ),
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
            cancellation_requested_at=None,
        )
    )
    await session.commit()
    return int(result.rowcount or 0)  # type: ignore[attr-defined]


async def claim_next_run(
    session: AsyncSession, *, worker_id: str, now: datetime | None = None
) -> MLRefinementRunRecord | None:
    timestamp = now or datetime.now(UTC)
    record = await session.scalar(
        select(MLRefinementRunRecord)
        .where(MLRefinementRunRecord.status == "queued")
        .order_by(MLRefinementRunRecord.created_at, MLRefinementRunRecord.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if record is None:
        await session.rollback()
        return None
    record.status = "running"
    record.worker_id = worker_id
    record.claimed_at = timestamp
    record.started_at = timestamp
    record.heartbeat_at = timestamp
    await session.commit()
    await session.refresh(record)
    return record


async def heartbeat(session: AsyncSession, run_id: UUID, worker_id: str) -> bool:
    result = await session.execute(
        update(MLRefinementRunRecord)
        .where(
            MLRefinementRunRecord.id == run_id,
            MLRefinementRunRecord.status == "running",
            MLRefinementRunRecord.worker_id == worker_id,
        )
        .values(heartbeat_at=datetime.now(UTC))
    )
    await session.commit()
    return bool(result.rowcount)  # type: ignore[attr-defined]
