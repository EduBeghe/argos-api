import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.market.catalog import Instrument
from app.ml.contract import (
    MISSING_DATA_POLICY_VERSION,
    ML_EXPERIMENT_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
)

from .calculation import FoldEvaluation, ScreeningResult
from .config import (
    ML_MODEL_SCREENING_VERSION,
    ModelScreeningConfiguration,
    ScreeningEngine,
)
from .models import (
    MLModelScreeningEvaluationRecord,
    MLModelScreeningResultRecord,
    MLModelScreeningRunRecord,
)

METRIC_QUANTUM = Decimal("0.000000000000000001")


def _decimal(value: float | None) -> Decimal | None:
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError("non-finite screening value cannot be persisted")
    return Decimal(str(value)).quantize(METRIC_QUANTUM)


def experiment_key(
    *,
    dataset_fingerprint: str,
    instrument: Instrument,
    engines: tuple[ScreeningEngine, ...],
    durations: tuple[int, ...],
    configuration: ModelScreeningConfiguration,
    versions: dict[str, str],
) -> str:
    payload = {
        "screening_version": ML_MODEL_SCREENING_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "instrument": asdict(instrument),
        "engines": [item.value for item in engines],
        "durations": list(durations),
        "configuration": configuration.as_dict(),
        "versions": versions,
        "seed": configuration.seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def find_completed_run(
    session: AsyncSession, key: str
) -> MLModelScreeningRunRecord | None:
    return cast(
        MLModelScreeningRunRecord | None,
        await session.scalar(
            select(MLModelScreeningRunRecord)
            .where(
                MLModelScreeningRunRecord.experiment_key == key,
                MLModelScreeningRunRecord.status == "completed",
            )
            .order_by(MLModelScreeningRunRecord.completed_at.desc())
            .limit(1)
        ),
    )


async def create_run(
    *,
    session: AsyncSession,
    key: str,
    dataset_fingerprint: str,
    instrument: Instrument,
    engines: tuple[ScreeningEngine, ...],
    durations: tuple[int, ...],
    configuration: ModelScreeningConfiguration,
    versions: dict[str, str],
    now: datetime,
) -> UUID:
    run_id = uuid4()
    session.add(
        MLModelScreeningRunRecord(
            id=run_id,
            experiment_key=key,
            status="running",
            screening_version=ML_MODEL_SCREENING_VERSION,
            experiment_contract_version=ML_EXPERIMENT_CONTRACT_VERSION,
            variable_registry_version=VARIABLE_REGISTRY_VERSION,
            missing_data_policy_version=MISSING_DATA_POLICY_VERSION,
            dataset_fingerprint=dataset_fingerprint,
            asset=instrument.asset.value,
            provider=instrument.provider.value,
            market_type=instrument.market_type.value,
            interval=instrument.interval.value,
            requested_engines=[item.value for item in engines],
            requested_future_durations=list(durations),
            library_versions=versions,
            deterministic_seed=configuration.seed,
            configuration=configuration.as_dict(),
            started_at=now,
            completed_at=None,
            failure_type=None,
            failure_message=None,
            evaluations_computed=0,
            evaluations_reused=0,
            results_generated=0,
            pareto_retained_count=0,
            elapsed_seconds=None,
            created_at=now,
        )
    )
    await session.commit()
    return run_id


async def persist_evaluations(
    session: AsyncSession,
    run_id: UUID,
    evaluations: tuple[FoldEvaluation, ...],
    now: datetime,
) -> None:
    if not evaluations:
        return
    rows = []
    for item in evaluations:
        metric = item.metrics
        rows.append(
            {
                "id": uuid4(),
                "run_id": run_id,
                "engine": item.engine.value,
                "future_duration_seconds": item.future_duration_seconds,
                "fold_name": item.fold_name,
                "training_row_count": item.training_row_count,
                "validation_row_count": item.validation_row_count,
                "sample_count": metric.sample_count,
                "mean_absolute_error": _decimal(metric.mean_absolute_error),
                "root_mean_squared_error": _decimal(metric.root_mean_squared_error),
                "mean_prediction": _decimal(metric.mean_prediction),
                "mean_actual_return": _decimal(metric.mean_actual_return),
                "pearson_correlation": _decimal(metric.pearson_correlation),
                "spearman_correlation": _decimal(metric.spearman_correlation),
                "directional_accuracy": _decimal(metric.directional_accuracy),
                "zero_prediction_mae": _decimal(metric.zero_prediction_mae),
                "mae_improvement_over_zero": _decimal(metric.mae_improvement_over_zero),
                "positive_prediction_count": metric.positive_prediction_count,
                "negative_prediction_count": metric.negative_prediction_count,
                "elapsed_seconds": _decimal(item.elapsed_seconds),
                "created_at": now,
            }
        )
    await session.execute(insert(MLModelScreeningEvaluationRecord), rows)
    await session.commit()


async def complete_run(
    session: AsyncSession,
    run_id: UUID,
    results: tuple[ScreeningResult, ...],
    evaluations: int,
    elapsed: float,
    now: datetime,
) -> None:
    rows = [
        {
            "id": uuid4(),
            "run_id": run_id,
            "engine": item.engine.value,
            "future_duration_seconds": item.future_duration_seconds,
            "folds_evaluated": item.folds_evaluated,
            "total_validation_samples": item.total_validation_samples,
            "median_mae_improvement_over_zero": _decimal(
                item.median_mae_improvement_over_zero
            ),
            "worst_fold_mae_improvement_over_zero": _decimal(
                item.worst_fold_mae_improvement_over_zero
            ),
            "median_directional_accuracy": _decimal(item.median_directional_accuracy),
            "worst_fold_directional_accuracy": _decimal(
                item.worst_fold_directional_accuracy
            ),
            "median_pearson_correlation": _decimal(item.median_pearson_correlation),
            "median_spearman_correlation": _decimal(item.median_spearman_correlation),
            "folds_beating_zero": item.folds_beating_zero,
            "folds_with_positive_spearman": item.folds_with_positive_spearman,
            "total_training_time_seconds": _decimal(item.total_training_time_seconds),
            "pareto_retained": item.pareto_retained,
            "display_rank": item.display_rank,
            "created_at": now,
        }
        for item in results
    ]
    if rows:
        await session.execute(insert(MLModelScreeningResultRecord), rows)
    await session.execute(
        update(MLModelScreeningRunRecord)
        .where(
            MLModelScreeningRunRecord.id == run_id,
            MLModelScreeningRunRecord.status == "running",
        )
        .values(
            status="completed",
            completed_at=now,
            evaluations_computed=evaluations,
            results_generated=len(results),
            pareto_retained_count=sum(item.pareto_retained for item in results),
            elapsed_seconds=_decimal(elapsed),
        )
    )
    await session.commit()


async def fail_run(
    session: AsyncSession, run_id: UUID, error: Exception, now: datetime
) -> None:
    await session.rollback()
    message = " ".join(str(error).split())[:512] or "screening failed"
    await session.execute(
        update(MLModelScreeningRunRecord)
        .where(
            MLModelScreeningRunRecord.id == run_id,
            MLModelScreeningRunRecord.status == "running",
        )
        .values(
            status="failed",
            completed_at=now,
            failure_type=type(error).__name__[:128],
            failure_message=message,
        )
    )
    await session.commit()
