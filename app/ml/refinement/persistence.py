import hashlib
import json
import logging
import math
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.contract import MISSING_DATA_POLICY_VERSION, VARIABLE_REGISTRY_VERSION
from app.ml.dataset import MLExperimentDataset
from app.ml.protected.contracts import (
    TEMPORAL_MANIFEST_VERSION,
    canonicalize_confidence_bucket,
)
from app.ml.protected.models import MLTemporalManifestRecord
from app.ml.protected.temporal import reconstruct_temporal_manifest
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
from .economic_policy_contract import economic_policy_evidence_compatibility
from .models import MLRefinementEvaluationRecord, MLRefinementRunRecord
from .planning import RefinementPlan

METRIC_QUANTUM = Decimal("0.000000000000000001")
logger = logging.getLogger(__name__)


class MLRefinementError(RuntimeError):
    pass


class SourceScreeningRunNotFoundError(MLRefinementError):
    pass


class SourceScreeningResultNotFoundError(MLRefinementError):
    pass


class SourceScreeningMismatchError(MLRefinementError):
    pass


class SourceScreeningIncompleteError(MLRefinementError):
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
    if run is None:
        raise SourceScreeningRunNotFoundError("source screening run does not exist")
    if result is None:
        raise SourceScreeningResultNotFoundError(
            "source screening result does not exist"
        )
    if result.run_id != run.id:
        raise SourceScreeningMismatchError(
            "source screening result does not belong to source run"
        )
    if run.status != "completed" or run.completed_at is None:
        raise SourceScreeningIncompleteError("source screening run is incomplete")
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
    session: AsyncSession,
    run: MLRefinementRunRecord,
    request: MLRefinementRequest | None = None,
) -> bool:
    count = await session.scalar(
        select(func.count(MLRefinementEvaluationRecord.id)).where(
            MLRefinementEvaluationRecord.run_id == run.id
        )
    )
    count_complete = (
        count == run.evaluations_selected
        and run.evaluations_computed + run.evaluations_reused
        == run.evaluations_selected
    )
    if not count_complete or request is None:
        return count_complete
    return not await incompatible_run_evaluations(session, run, request)


async def incompatible_run_evaluations(
    session: AsyncSession,
    run: MLRefinementRunRecord,
    request: MLRefinementRequest,
) -> tuple[MLRefinementEvaluationRecord, ...]:
    rows = tuple(
        (
            await session.scalars(
                select(MLRefinementEvaluationRecord)
                .where(MLRefinementEvaluationRecord.run_id == run.id)
                .order_by(
                    MLRefinementEvaluationRecord.created_at,
                    MLRefinementEvaluationRecord.id,
                )
            )
        ).all()
    )
    return tuple(
        row
        for row in rows
        if not economic_policy_evidence_compatibility(
            aggregate_metrics=row.aggregate_metrics,
            request=request,
            future_duration_seconds=row.future_duration_seconds,
        ).compatible
    )


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
        compatible_fingerprints = [
            effective_evaluation_fingerprint(
                dataset_fingerprint=dataset_fingerprint,
                candidate_fingerprint=candidate.fingerprint,
                engine=candidate.engine.value,
                versions=versions,
                deterministic_seed=request.deterministic_seed,
            )
            for candidate in plan.evaluations
        ]
        await session.execute(
            delete(MLRefinementEvaluationRecord).where(
                MLRefinementEvaluationRecord.run_id == existing.id,
                MLRefinementEvaluationRecord.fingerprint.not_in(
                    compatible_fingerprints
                ),
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
                evaluations_selected=len(plan.evaluations),
                budget_exhausted=plan.omitted_by_trial_budget > 0,
                termination_reason=None,
                elapsed_seconds=None,
                worker_id="direct-cli",
                claimed_at=now,
                heartbeat_at=now,
                cancellation_requested_at=None,
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
        evaluations_selected=len(plan.evaluations),
        budget_exhausted=plan.omitted_by_trial_budget > 0,
        termination_reason=None,
        started_at=now,
        completed_at=None,
        failed_at=None,
        failure_type=None,
        failure_message=None,
        elapsed_seconds=None,
        worker_id="direct-cli",
        claimed_at=now,
        heartbeat_at=now,
        cancellation_requested_at=None,
        created_at=now,
    )
    session.add(record)
    await session.commit()
    return record


def effective_evaluation_fingerprint(
    *,
    dataset_fingerprint: str,
    candidate_fingerprint: str,
    engine: str,
    versions: dict[str, str],
    deterministic_seed: int,
) -> str:
    engine_version_keys = {
        "ridge": ("scikit-learn",),
        "lightgbm": ("lightgbm",),
        "catboost": ("catboost",),
    }
    relevant_versions = {
        key: versions[key] for key in engine_version_keys[engine] if key in versions
    }
    payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "candidate_fingerprint": candidate_fingerprint,
        "contract_version": ML_REFINEMENT_CONTRACT_VERSION,
        "classification_version": ML_REFINEMENT_CLASSIFICATION_VERSION,
        "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "missing_data_policy_version": MISSING_DATA_POLICY_VERSION,
        "engine_versions": relevant_versions,
        "deterministic_seed": deterministic_seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def existing_run_evaluation(
    session: AsyncSession,
    run_id: UUID,
    fingerprint: str,
    request: MLRefinementRequest | None = None,
) -> MLRefinementEvaluationRecord | None:
    row = cast(
        MLRefinementEvaluationRecord | None,
        await session.scalar(
            select(MLRefinementEvaluationRecord).where(
                MLRefinementEvaluationRecord.run_id == run_id,
                MLRefinementEvaluationRecord.fingerprint == fingerprint,
            )
        ),
    )
    return _compatible_reuse(row, request=request, run_id=run_id)


async def reusable_evaluation(
    session: AsyncSession,
    fingerprint: str,
    request: MLRefinementRequest | None = None,
) -> MLRefinementEvaluationRecord | None:
    rows = tuple(
        (
            await session.scalars(
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
                .limit(100)
            )
        ).all()
    )
    for row in rows:
        compatible = _compatible_reuse(row, request=request, run_id=row.run_id)
        if compatible is not None:
            return compatible
    return None


def _compatible_reuse(
    row: MLRefinementEvaluationRecord | None,
    *,
    request: MLRefinementRequest | None,
    run_id: UUID,
) -> MLRefinementEvaluationRecord | None:
    if row is None or request is None:
        return row
    compatibility = economic_policy_evidence_compatibility(
        aggregate_metrics=row.aggregate_metrics,
        request=request,
        future_duration_seconds=row.future_duration_seconds,
    )
    if compatibility.compatible:
        return row
    logger.warning(
        "Rejected semantically incompatible ML refinement reuse",
        extra={
            "structured_fields": {
                "run_id": str(run_id),
                "evaluation_id": str(row.id),
                "reason_code": compatibility.reason_code,
                "expected_contract_fingerprint": compatibility.expected_contract,
                "found_contract_fingerprint": compatibility.found_contract,
            }
        },
    )
    return None


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return str(decimal_metric(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_fraction(value: object, configured: list[str]) -> str:
    persisted = str(decimal_metric(value)) if isinstance(value, float) else value
    return str(canonicalize_confidence_bucket(persisted, configured))


def _canonical_confidence_metrics(metrics: object, configured: list[str]) -> object:
    if not isinstance(metrics, (list, tuple)):
        return _json_safe(metrics)
    return [
        {
            **{
                str(key): _json_safe(value)
                for key, value in item.items()
                if key != "fraction"
            },
            "fraction": _canonical_fraction(item.get("fraction"), configured),
        }
        if isinstance(item, dict)
        else _json_safe(item)
        for item in metrics
    ]


MAX_PERSISTED_AGGREGATE_METRICS_BYTES = 1_000_000


class MLRefinementEvidenceTooLargeError(ValueError):
    """Raised before persistence when compact evidence exceeds its hard limit."""


def _canonical_aggregate_metrics(
    metrics: dict[str, object],
    configured: list[str],
    *,
    canonicalize_confidence: bool = True,
) -> dict[str, object]:
    persisted = cast(dict[str, object], _json_safe(metrics))
    economic_aggregates = persisted.get("economic_policy_v2_aggregates")
    if isinstance(economic_aggregates, list):
        persisted["economic_policy_v2_aggregates"] = [
            {str(key): value for key, value in item.items() if key != "folds"}
            if isinstance(item, dict)
            else item
            for item in economic_aggregates
        ]
    selected_policy = persisted.get("selected_economic_policy_v2")
    if isinstance(selected_policy, dict):
        persisted["selected_economic_policy_v2"] = {
            str(key): value for key, value in selected_policy.items() if key != "folds"
        }
    if canonicalize_confidence:
        aggregates = metrics.get("confidence_bucket_aggregates")
        if aggregates is not None:
            persisted["confidence_bucket_aggregates"] = _canonical_confidence_metrics(
                aggregates, configured
            )
        selected = metrics.get("selected_confidence_bucket")
        if isinstance(selected, dict):
            persisted["selected_confidence_bucket"] = {
                **cast(dict[str, object], _json_safe(selected)),
                "fraction": _canonical_fraction(selected.get("fraction"), configured),
            }
    encoded_size = len(
        json.dumps(persisted, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if encoded_size > MAX_PERSISTED_AGGREGATE_METRICS_BYTES:
        raise MLRefinementEvidenceTooLargeError(
            "compacted aggregate metrics exceed "
            f"{MAX_PERSISTED_AGGREGATE_METRICS_BYTES} bytes: {encoded_size}"
        )
    return persisted


async def persist_evaluation(
    *,
    session: AsyncSession,
    run_id: UUID,
    fingerprint: str,
    result: RefinementEvaluationResult | None,
    reused: MLRefinementEvaluationRecord | None,
    request: MLRefinementRequest,
    now: datetime,
    development_dataset: MLExperimentDataset | None = None,
    canonical_dataset_fingerprint: str | None = None,
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
        values["aggregate_metrics"] = _canonical_aggregate_metrics(
            reused.aggregate_metrics,
            [str(item) for item in request.confidence_buckets],
            canonicalize_confidence=False,
        )
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
    configured_buckets = [str(item) for item in request.confidence_buckets]
    folds = [
        {
            **asdict(fold),
            "metrics": _json_safe(fold.metrics),
            "confidence_metrics": _canonical_confidence_metrics(
                fold.confidence_metrics, configured_buckets
            ),
        }
        for fold in result.folds
    ]
    record = MLRefinementEvaluationRecord(
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
        confidence_buckets=configured_buckets,
        fold_evidence=folds,
        aggregate_metrics=_canonical_aggregate_metrics(
            result.aggregate_metrics, configured_buckets
        ),
        classification=result.classification.value,
        classification_version=ML_REFINEMENT_CLASSIFICATION_VERSION,
        failed_rules=list(result.failed_rules),
        elapsed_seconds=decimal_metric(result.elapsed_seconds),
        created_at=now,
    )
    session.add(record)
    # The temporal manifest and any future evaluation evidence reference this row.
    # Flush the parent explicitly so correctness does not depend on SQLAlchemy's
    # incidental ordering for independently mapped records with assigned UUIDs.
    await session.flush()
    if (
        development_dataset is not None
        and canonical_dataset_fingerprint is not None
        and len(result.folds) == len(development_dataset.folds)
    ):
        manifest = reconstruct_temporal_manifest(
            dataset=development_dataset,
            evaluation=record,
            canonical_dataset_fingerprint=canonical_dataset_fingerprint,
        )
        session.add(
            MLTemporalManifestRecord(
                id=uuid4(),
                evaluation_id=record.id,
                source="persisted_at_execution",
                status="completed",
                manifest_version=TEMPORAL_MANIFEST_VERSION,
                split_policy_version=TEMPORAL_MANIFEST_VERSION,
                dataset_fingerprint=canonical_dataset_fingerprint,
                manifest_fingerprint=manifest.fingerprint,
                manifest=manifest.payload,
                fingerprint_verified=True,
                counts_verified=True,
                created_at=now,
                completed_at=now,
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
    worker_id: str | None = None,
) -> None:
    statement = update(MLRefinementRunRecord).where(
        MLRefinementRunRecord.id == run_id,
        MLRefinementRunRecord.status == "running",
    )
    if worker_id is not None:
        statement = statement.where(MLRefinementRunRecord.worker_id == worker_id)
    result = await session.execute(
        statement.values(
            status="completed",
            completed_at=now,
            evaluations_computed=computed,
            evaluations_reused=reused,
            evaluations_omitted=omitted,
            budget_exhausted=budget_exhausted,
            termination_reason=reason,
            elapsed_seconds=decimal_metric(elapsed),
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
            cancellation_requested_at=None,
        )
    )
    if not cast(CursorResult[object], result).rowcount:
        await session.rollback()
        raise MLRefinementError("worker lost the refinement lease")
    await session.commit()


async def record_progress(
    session: AsyncSession,
    run_id: UUID,
    *,
    computed: int,
    reused: int,
    worker_id: str | None = None,
) -> None:
    statement = update(MLRefinementRunRecord).where(
        MLRefinementRunRecord.id == run_id,
        MLRefinementRunRecord.status == "running",
    )
    if worker_id is not None:
        statement = statement.where(MLRefinementRunRecord.worker_id == worker_id)
    result = await session.execute(
        statement.values(
            evaluations_computed=computed,
            evaluations_reused=reused,
            heartbeat_at=datetime.now(UTC),
        )
    )
    if not cast(CursorResult[object], result).rowcount:
        await session.rollback()
        raise MLRefinementError("worker lost the refinement lease")
    await session.commit()


async def fail_run(
    session: AsyncSession,
    run_id: UUID,
    error: Exception,
    now: datetime,
    *,
    worker_id: str | None = None,
) -> None:
    await session.rollback()
    statement = update(MLRefinementRunRecord).where(
        MLRefinementRunRecord.id == run_id,
        MLRefinementRunRecord.status == "running",
    )
    if worker_id is not None:
        statement = statement.where(MLRefinementRunRecord.worker_id == worker_id)
    await session.execute(
        statement.values(
            status="failed",
            failed_at=now,
            failure_type=type(error).__name__[:128],
            failure_message=(" ".join(str(error).split()) or "refinement failed")[:512],
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
            cancellation_requested_at=None,
        )
    )
    await session.commit()


async def cancel_claimed_run(
    session: AsyncSession, run_id: UUID, *, worker_id: str, now: datetime
) -> None:
    result = await session.execute(
        update(MLRefinementRunRecord)
        .where(
            MLRefinementRunRecord.id == run_id,
            MLRefinementRunRecord.status == "running",
            MLRefinementRunRecord.worker_id == worker_id,
        )
        .values(
            status="cancelled",
            completed_at=now,
            termination_reason="cancelled_by_user",
            worker_id=None,
            claimed_at=None,
            heartbeat_at=None,
            cancellation_requested_at=None,
        )
    )
    if not cast(CursorResult[object], result).rowcount:
        await session.rollback()
        raise MLRefinementError("worker lost the refinement lease")
    await session.commit()
