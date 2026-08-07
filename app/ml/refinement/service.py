import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.dataset import MLDatasetMetadata, MLExperimentDataset
from app.ml.screening.engines import library_versions

from .calculation import RefinementEvaluationResult, evaluate_refinement_candidate
from .contracts import MLRefinementRequest
from .identity import validate_dataset_identity
from .models import MLRefinementRunRecord
from .persistence import (
    cancel_claimed_run,
    complete_run,
    completed_run_is_complete,
    effective_evaluation_fingerprint,
    existing_run_evaluation,
    experiment_key,
    fail_run,
    find_completed_run,
    persist_evaluation,
    record_progress,
    resolve_source,
    reusable_evaluation,
    start_run,
)
from .planning import PlannedEvaluation, RefinementPlan, build_refinement_plan
from .queue import heartbeat

logger = logging.getLogger(__name__)


class RefinementLeaseLostError(RuntimeError):
    pass


async def _evaluate_with_heartbeat(
    *,
    session: AsyncSession,
    run_id: UUID,
    worker_id: str,
    heartbeat_interval_seconds: float,
    dataset: MLExperimentDataset,
    candidate: PlannedEvaluation,
    request: MLRefinementRequest,
) -> RefinementEvaluationResult:
    stopped = asyncio.Event()
    lease_error: Exception | None = None

    async def maintain_lease() -> None:
        nonlocal lease_error
        try:
            while not stopped.is_set():
                try:
                    await asyncio.wait_for(
                        stopped.wait(), timeout=heartbeat_interval_seconds
                    )
                except TimeoutError:
                    if not await heartbeat(session, run_id, worker_id):
                        raise RefinementLeaseLostError(
                            "worker lost the refinement lease"
                        ) from None
        except Exception as error:
            lease_error = error

    task = asyncio.create_task(maintain_lease())
    try:
        result = await asyncio.to_thread(
            evaluate_refinement_candidate,
            dataset=dataset,
            candidate=candidate,
            request=request,
        )
    finally:
        stopped.set()
        await task
    if lease_error is not None:
        raise RefinementLeaseLostError(
            "worker lease could not be verified during evaluation"
        ) from lease_error
    if not await heartbeat(session, run_id, worker_id):
        raise RefinementLeaseLostError("worker lost the refinement lease")
    return result


@dataclass(frozen=True, slots=True)
class RefinementExecutionResult:
    run_id: UUID
    experiment_key: str
    reused_existing_run: bool
    combinations_planned: int
    evaluations_computed: int
    evaluations_reused: int
    evaluations_omitted: int
    budget_exhausted: bool
    termination_reason: str


@dataclass(frozen=True, slots=True)
class RefinementDryRun:
    experiment_key: str
    plan: RefinementPlan
    fold_count: int


async def prepare_refinement(
    *,
    session: AsyncSession,
    request: MLRefinementRequest,
    dataset_fingerprint: str,
) -> RefinementDryRun:
    await resolve_source(session, request)
    plan = build_refinement_plan(request)
    return RefinementDryRun(
        experiment_key(
            request=request,
            dataset_fingerprint=dataset_fingerprint,
            versions=library_versions(),
        ),
        plan,
        7,
    )


async def execute_refinement(
    *,
    session: AsyncSession,
    identity_metadata: MLDatasetMetadata,
    development_dataset: MLExperimentDataset,
    request: MLRefinementRequest,
    now: datetime | None = None,
    claimed_run: MLRefinementRunRecord | None = None,
    worker_id: str | None = None,
    heartbeat_interval_seconds: float | None = None,
) -> RefinementExecutionResult:
    source_run, _ = await resolve_source(session, request)
    versions = library_versions()
    key = experiment_key(
        request=request,
        dataset_fingerprint=identity_metadata.fingerprint,
        versions=versions,
    )
    existing = await find_completed_run(session, key)
    if (
        claimed_run is None
        and existing is not None
        and await completed_run_is_complete(session, existing, request)
    ):
        return RefinementExecutionResult(
            existing.id,
            key,
            True,
            existing.combinations_planned,
            0,
            existing.evaluations_computed + existing.evaluations_reused,
            existing.evaluations_omitted,
            existing.budget_exhausted,
            existing.termination_reason or "completed",
        )
    plan = build_refinement_plan(request)
    timestamp = now or datetime.now(UTC)
    if claimed_run is not None:
        if worker_id is None or claimed_run.worker_id != worker_id:
            raise RuntimeError("claimed refinement worker does not own the lease")
        diagnostics = validate_dataset_identity(
            source=source_run, queued=claimed_run, rebuilt=identity_metadata
        )
        if claimed_run.experiment_key != key:
            raise RuntimeError("claimed refinement experiment identity changed")
        run = claimed_run
        if heartbeat_interval_seconds is None or heartbeat_interval_seconds <= 0:
            raise ValueError("claimed execution requires a positive heartbeat interval")
    else:
        run = await start_run(
            session=session,
            request=request,
            plan=plan,
            key=key,
            dataset_fingerprint=identity_metadata.fingerprint,
            versions=versions,
            now=timestamp,
        )
    structured_fields: dict[str, object] = {
        "run_id": str(run.id),
        "experiment_key": key,
        "planned": plan.total_combinations,
        "selected": len(plan.evaluations),
        "development_start": development_dataset.metadata.requested_start.isoformat(),
        "development_end_exclusive": (
            development_dataset.metadata.requested_end.isoformat()
        ),
        "development_candle_count": development_dataset.metadata.candle_count,
    }
    if claimed_run is not None:
        structured_fields.update(diagnostics.structured_fields())
        structured_fields["retry_start"] = run.evaluations_computed == 0
    logger.info(
        "ML refinement started",
        extra={"structured_fields": structured_fields},
    )
    started = perf_counter()
    computed = 0
    reused_count = 0
    omitted = plan.omitted_by_trial_budget
    reason = "trial_plan_completed"
    try:
        for index, candidate in enumerate(plan.evaluations):
            if worker_id is not None:
                await session.refresh(run)
                if run.cancellation_requested_at is not None:
                    await cancel_claimed_run(
                        session, run.id, worker_id=worker_id, now=datetime.now(UTC)
                    )
                    return RefinementExecutionResult(
                        run.id,
                        key,
                        False,
                        plan.total_combinations,
                        computed,
                        reused_count,
                        omitted + len(plan.evaluations) - index,
                        True,
                        "cancelled_by_user",
                    )
                if not await heartbeat(session, run.id, worker_id):
                    raise RuntimeError("worker lost the refinement lease")
            if perf_counter() - started >= request.maximum_runtime_seconds:
                omitted += len(plan.evaluations) - index
                reason = "maximum_runtime_seconds"
                break
            fingerprint = effective_evaluation_fingerprint(
                dataset_fingerprint=identity_metadata.fingerprint,
                candidate_fingerprint=candidate.fingerprint,
                engine=candidate.engine.value,
                versions=versions,
                deterministic_seed=request.deterministic_seed,
            )
            retained = await existing_run_evaluation(
                session, run.id, fingerprint, request
            )
            if retained is not None:
                reused_count += 1
                await record_progress(
                    session,
                    run.id,
                    computed=computed,
                    reused=reused_count,
                    worker_id=worker_id,
                )
                continue
            reused = await reusable_evaluation(session, fingerprint, request)
            if reused is not None:
                await persist_evaluation(
                    session=session,
                    run_id=run.id,
                    fingerprint=fingerprint,
                    result=None,
                    reused=reused,
                    request=request,
                    now=datetime.now(UTC),
                )
                reused_count += 1
            else:
                if worker_id is None:
                    result = evaluate_refinement_candidate(
                        dataset=development_dataset,
                        candidate=candidate,
                        request=request,
                    )
                else:
                    assert heartbeat_interval_seconds is not None
                    result = await _evaluate_with_heartbeat(
                        session=session,
                        run_id=run.id,
                        worker_id=worker_id,
                        heartbeat_interval_seconds=heartbeat_interval_seconds,
                        dataset=development_dataset,
                        candidate=candidate,
                        request=request,
                    )
                    await session.refresh(run)
                    if run.cancellation_requested_at is not None:
                        await cancel_claimed_run(
                            session,
                            run.id,
                            worker_id=worker_id,
                            now=datetime.now(UTC),
                        )
                        return RefinementExecutionResult(
                            run.id,
                            key,
                            False,
                            plan.total_combinations,
                            computed,
                            reused_count,
                            omitted + len(plan.evaluations) - index,
                            True,
                            "cancelled_by_user",
                        )
                await persist_evaluation(
                    session=session,
                    run_id=run.id,
                    fingerprint=fingerprint,
                    result=result,
                    reused=None,
                    request=request,
                    now=datetime.now(UTC),
                    development_dataset=development_dataset,
                    canonical_dataset_fingerprint=identity_metadata.fingerprint,
                )
                computed += 1
            if worker_id is not None and not await heartbeat(
                session, run.id, worker_id
            ):
                raise RuntimeError("worker lost the refinement lease")
            await record_progress(
                session,
                run.id,
                computed=computed,
                reused=reused_count,
                worker_id=worker_id,
            )
            if (index + 1) % 10 == 0 or index + 1 == len(plan.evaluations):
                logger.info(
                    "ML refinement progress",
                    extra={
                        "structured_fields": {
                            "run_id": str(run.id),
                            "completed": index + 1,
                            "computed": computed,
                            "reused": reused_count,
                        }
                    },
                )
        elapsed = perf_counter() - started
        budget_exhausted = omitted > 0
        await complete_run(
            session,
            run.id,
            computed=computed,
            reused=reused_count,
            omitted=omitted,
            budget_exhausted=budget_exhausted,
            reason=reason,
            elapsed=elapsed,
            now=datetime.now(UTC),
            worker_id=worker_id,
        )
    except Exception as error:
        if worker_id is None:
            await fail_run(session, run.id, error, datetime.now(UTC))
            logger.exception(
                "ML refinement failed",
                extra={
                    "structured_fields": {
                        "run_id": str(run.id),
                        "failure_type": type(error).__name__,
                    }
                },
            )
        raise
    logger.info(
        "ML refinement completed",
        extra={
            "structured_fields": {
                "run_id": str(run.id),
                "computed": computed,
                "reused": reused_count,
                "omitted": omitted,
                "termination_reason": reason,
            }
        },
    )
    return RefinementExecutionResult(
        run.id,
        key,
        False,
        plan.total_combinations,
        computed,
        reused_count,
        omitted,
        omitted > 0,
        reason,
    )

