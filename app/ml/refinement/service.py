import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.dataset import MLExperimentDataset
from app.ml.screening.engines import library_versions

from .calculation import evaluate_refinement_candidate
from .contracts import MLRefinementRequest
from .persistence import (
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
from .planning import RefinementPlan, build_refinement_plan

logger = logging.getLogger(__name__)


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
    dataset: MLExperimentDataset,
    request: MLRefinementRequest,
    now: datetime | None = None,
) -> RefinementExecutionResult:
    await resolve_source(session, request)
    versions = library_versions()
    key = experiment_key(
        request=request,
        dataset_fingerprint=dataset.metadata.fingerprint,
        versions=versions,
    )
    existing = await find_completed_run(session, key)
    if existing is not None and await completed_run_is_complete(session, existing):
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
    run = await start_run(
        session=session,
        request=request,
        plan=plan,
        key=key,
        dataset_fingerprint=dataset.metadata.fingerprint,
        versions=versions,
        now=timestamp,
    )
    logger.info(
        "ML refinement started",
        extra={
            "structured_fields": {
                "run_id": str(run.id),
                "experiment_key": key,
                "planned": plan.total_combinations,
                "selected": len(plan.evaluations),
            }
        },
    )
    started = perf_counter()
    computed = 0
    reused_count = 0
    omitted = plan.omitted_by_trial_budget
    reason = "trial_plan_completed"
    try:
        for index, candidate in enumerate(plan.evaluations):
            if perf_counter() - started >= request.maximum_runtime_seconds:
                omitted += len(plan.evaluations) - index
                reason = "maximum_runtime_seconds"
                break
            fingerprint = effective_evaluation_fingerprint(
                dataset_fingerprint=dataset.metadata.fingerprint,
                candidate_fingerprint=candidate.fingerprint,
                engine=candidate.engine.value,
                versions=versions,
                deterministic_seed=request.deterministic_seed,
            )
            retained = await existing_run_evaluation(session, run.id, fingerprint)
            if retained is not None:
                reused_count += 1
                await record_progress(
                    session, run.id, computed=computed, reused=reused_count
                )
                continue
            reused = await reusable_evaluation(session, fingerprint)
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
                result = evaluate_refinement_candidate(
                    dataset=dataset, candidate=candidate, request=request
                )
                await persist_evaluation(
                    session=session,
                    run_id=run.id,
                    fingerprint=fingerprint,
                    result=result,
                    reused=None,
                    request=request,
                    now=datetime.now(UTC),
                )
                computed += 1
            await record_progress(
                session, run.id, computed=computed, reused=reused_count
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
        )
    except Exception as error:
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
