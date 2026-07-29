
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.contract import MISSING_DATA_POLICY_VERSION, VARIABLE_REGISTRY_VERSION
from app.ml.screening.models import (
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)

from .calculation import RefinementEvaluationResult
from .contracts import (
    ML_REFINEMENT_CLASSIFICATION_VERSION,
    ML_REFINEMENT_CONTRACT_VERSION,
    MLRefinementRequest,
)
from .models import MLRefinementEvaluationRecord, MLRefinementRunRecord
from .planning import RefinementPlan

METRIC_QUANTUM = Decimal("0.000000000000000001")


class MLRefinementError(RuntimeError):
    pass


def decimal_metric(value: float) -> Decimal:
    if not math.isfinite(value):
        raise MLRefinementError("non-finite refinement metric cannot be persisted")
    return Decimal(str(value)).quantize(METRIC_QUANTUM)


def experiment_key(
    *, request: MLRefinementRequest, dataset_fingerprint: str, versions: dict[str, str]
) -> str:
    payload = {
        "contract_version": ML_REFINEMENT_CONTRACT_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "source_screening_result_id": str(request.source_screening_result_id),
        "configuration": request.normalized_payload(),
        "versions": versions,
        "seed": request.deterministic_seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def resolve_source(
    session: AsyncSession, request: MLRefinementRequest
) -> tuple[MLModelScreeningRunRecord, MLModelScreeningResultRecord]:
    run = await session.get(MLModelScreeningRunRecord, request.source_screening_run_id)
    result = await session.get(
        MLModelScreeningResultRecord, request.source_screening_result_id
    )
    if run is None or result is None:
        raise MLRefinementError("source screening run or result does not exist")
    if result.run_id != run.id:
        raise MLRefinementError("source screening result does not belong to source run")
    if run.status != "completed" or run.completed_at is None:
        raise MLRefinementError("source screening run is incomplete")
    return run, result


async def find_completed_run(
    session: AsyncSession, key: str
) -> MLRefinementRunRecord | None:
    return cast(
        MLRefinementRunRecord | None,
        await session.scalar(
            select(MLRefinementRunRecord).where(
                MLRefinementRunRecord.experiment_key == key,
                MLRefinementRunRecord.status == "completed",
            )
        ),
    )


async def completed_run_is_complete(
    session: AsyncSession, run: MLRefinementRunRecord
) -> bool:
    count = await session.scalar(
        select(func.count(MLRefinementEvaluationRecord.id)).where(
            MLRefinementEvaluationRecord.run_id == run.id
        )
    )
    return count == run.evaluations_computed + run.evaluations_reused


async def start_run(
    *,
    session: AsyncSession,
    request: MLRefinementRequest,
    plan: RefinementPlan,
    key: str,
    dataset_fingerprint: str,
    versions: dict[str, str],
    now: datetime,
) -> MLRefinementRunRecord:
    existing = await session.scalar(
        select(MLRefinementRunRecord).where(MLRefinementRunRecord.experiment_key == key)
    )
    if existing is not None:
        if existing.status in {"queued", "running"}:
            raise MLRefinementError("an identical refinement is already running")
        await session.execute(
            delete(MLRefinementEvaluationRecord).where(
                MLRefinementEvaluationRecord.run_id == existing.id
            )
        )
        await session.execute(
            update(MLRefinementRunRecord)
            .where(MLRefinementRunRecord.id == existing.id)
            .values(
                status="running",
                started_at=now,
                completed_at=None,
                failed_at=None,
                failure_type=None,
                failure_message=None,
                evaluations_computed=0,
                evaluations_reused=0,
                evaluations_omitted=plan.omitted_by_trial_budget,
                budget_exhausted=plan.omitted_by_trial_budget > 0,
                termination_reason=None,
                elapsed_seconds=None,
            )
        )
        await session.commit()
        await session.refresh(existing)
        return existing
    record = MLRefinementRunRecord(
        id=uuid4(),
        experiment_key=key,
        source_screening_run_id=request.source_screening_run_id,
        source_screening_result_id=request.source_screening_result_id,
        status="running",
        contract_version=ML_REFINEMENT_CONTRACT_VERSION,
        classification_version=ML_REFINEMENT_CLASSIFICATION_VERSION,
        dataset_fingerprint=dataset_fingerprint,
        variable_registry_version=VARIABLE_REGISTRY_VERSION,
        missing_data_policy_version=MISSING_DATA_POLICY_VERSION,
        library_versions=versions,
        normalized_configuration=request.normalized_payload(),
        notes=request.notes,
        maximum_trials=request.maximum_trials,
        maximum_runtime_seconds=request.maximum_runtime_seconds,
        combinations_planned=plan.total_combinations,
        evaluations_computed=0,
        evaluations_reused=0,
        evaluations_omitted=plan.omitted_by_trial_budget,
        budget_exhausted=plan.omitted_by_trial_budget > 0,
        termination_reason=None,
        started_at=now,
        completed_at=None,
        failed_at=None,
        failure_type=None,
        failure_message=None,
        elapsed_seconds=None,
        created_at=now,
    )
    session.add(record)
    await session.commit()
    return record


def effective_evaluation_fingerprint(
    dataset_fingerprint: str, candidate_fingerprint: str
) -> str:
    return hashlib.sha256(
        f"{dataset_fingerprint}:{candidate_fingerprint}".encode()
    ).hexdigest()


async def reusable_evaluation(
    session: AsyncSession, fingerprint: str
) -> MLRefinementEvaluationRecord | None:
    return cast(
        MLRefinementEvaluationRecord | None,
        await session.scalar(
            select(MLRefinementEvaluationRecord)
            .join(
                MLRefinementRunRecord,
                MLRefinementRunRecord.id == MLRefinementEvaluationRecord.run_id,
            )
            .where(
                MLRefinementEvaluationRecord.fingerprint == fingerprint,
                MLRefinementRunRecord.status == "completed",
            )
            .order_by(MLRefinementEvaluationRecord.created_at.desc())
            .limit(1)
        ),
    )


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return str(decimal_metric(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


async def persist_evaluation(
    *,
    session: AsyncSession,
    run_id: UUID,
    fingerprint: str,
    result: RefinementEvaluationResult | None,
    reused: MLRefinementEvaluationRecord | None,
    request: MLRefinementRequest,
    now: datetime,
) -> None:
    if (result is None) == (reused is None):
        raise ValueError("exactly one evaluation source is required")
    if reused is not None:
        values = {
            column.name: getattr(reused, column.name)
            for column in reused.__table__.columns
            if column.name
            not in {"id", "run_id", "reused_from_evaluation_id", "created_at"}
        }
        values.update(
            id=uuid4(),
            run_id=run_id,
            reused_from_evaluation_id=reused.id,
            created_at=now,
        )
        session.add(MLRefinementEvaluationRecord(**values))
        await session.commit()
        return
    assert result is not None
    candidate = result.candidate
    folds = [
        {
            **asdict(fold),
            "metrics": _json_safe(fold.metrics),
            "confidence_metrics": _json_safe(fold.confidence_metrics),
        }
        for fold in result.folds
    ]
    session.add(
        MLRefinementEvaluationRecord(
            id=uuid4(),
            run_id=run_id,
            reused_from_evaluation_id=None,
            fingerprint=fingerprint,
            engine=candidate.engine.value,
            future_duration_seconds=candidate.future_duration_seconds,
            objective=candidate.objective.value,
            hyperparameters=dict(candidate.hyperparameters),
            feature_policy={
                "strategy": candidate.feature_strategy,
                "variables": candidate.explicit_variables,
                "top_k": candidate.top_k,
            },
            confidence_buckets=[str(item) for item in request.confidence_buckets],
            fold_evidence=folds,
            aggregate_metrics=_json_safe(result.aggregate_metrics),
            classification=result.classification.value,
            classification_version=ML_REFINEMENT_CLASSIFICATION_VERSION,
            failed_rules=list(result.failed_rules),
            elapsed_seconds=decimal_metric(result.elapsed_seconds),
            created_at=now,
        )
    )
    await session.commit()


async def complete_run(
    session: AsyncSession,
    run_id: UUID,
    *,
    computed: int,
    reused: int,
    omitted: int,
    budget_exhausted: bool,
    reason: str,
    elapsed: float,
    now: datetime,
) -> None:
    await session.execute(
        update(MLRefinementRunRecord)
        .where(
            MLRefinementRunRecord.id == run_id,
            MLRefinementRunRecord.status == "running",
        )
        .values(
            status="completed",
            completed_at=now,
            evaluations_computed=computed,
            evaluations_reused=reused,
            evaluations_omitted=omitted,
            budget_exhausted=budget_exhausted,
            termination_reason=reason,
            elapsed_seconds=decimal_metric(elapsed),
        )
    )
    await session.commit()


async def record_progress(
    session: AsyncSession, run_id: UUID, *, computed: int, reused: int
) -> None:
    await session.execute(
        update(MLRefinementRunRecord)
        .where(
            MLRefinementRunRecord.id == run_id,
            MLRefinementRunRecord.status == "running",
        )
        .values(evaluations_computed=computed, evaluations_reused=reused)
    )
    await session.commit()


async def fail_run(
    session: AsyncSession, run_id: UUID, error: Exception, now: datetime
) -> None:
    await session.rollback()
    await session.execute(
        update(MLRefinementRunRecord)
        .where(
            MLRefinementRunRecord.id == run_id,
            MLRefinementRunRecord.status == "running",
        )
        .values(
            status="failed",
            failed_at=now,
            failure_type=type(error).__name__[:128],
            failure_message=(" ".join(str(error).split()) or "refinement failed")[:512],
        )
    )
    await session.commit()
