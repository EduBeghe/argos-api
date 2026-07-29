import logging
import tracemalloc
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.dataset import MLExperimentDataset

from .calculation import (
    ScreeningResult,
    aggregate_evaluations,
    apply_pareto,
    evaluate_combination,
)
from .config import ModelScreeningConfiguration, ScreeningEngine
from .engines import library_versions
from .persistence import (
    complete_run,
    create_run,
    experiment_key,
    fail_run,
    find_completed_run,
    persist_evaluations,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScreeningExecutionResult:
    run_id: UUID
    experiment_key: str
    reused_existing_run: bool
    evaluations_computed: int
    evaluations_reused: int
    results_generated: int
    pareto_retained_count: int


async def screen_models(
    *,
    session: AsyncSession,
    dataset: MLExperimentDataset,
    engines: tuple[ScreeningEngine, ...],
    durations: tuple[timedelta, ...],
    configuration: ModelScreeningConfiguration,
    force_recompute: bool = False,
    now: datetime | None = None,
) -> ScreeningExecutionResult:
    versions = library_versions()
    duration_seconds = tuple(int(item.total_seconds()) for item in durations)
    key = experiment_key(
        dataset_fingerprint=dataset.metadata.fingerprint,
        instrument=dataset.metadata.instrument,
        engines=engines,
        durations=duration_seconds,
        configuration=configuration,
        versions=versions,
    )
    if not force_recompute:
        existing = await find_completed_run(session, key)
        if existing is not None:
            return ScreeningExecutionResult(
                existing.id,
                key,
                True,
                0,
                existing.evaluations_computed,
                existing.results_generated,
                existing.pareto_retained_count,
            )
    timestamp = now or datetime.now(UTC)
    run_id = await create_run(
        session=session,
        key=key,
        dataset_fingerprint=dataset.metadata.fingerprint,
        instrument=dataset.metadata.instrument,
        engines=engines,
        durations=duration_seconds,
        configuration=configuration,
        versions=versions,
        now=timestamp,
    )
    logger.info(
        "ML model screening started",
        extra={
            "structured_fields": {
                "run_id": str(run_id),
                "experiment_key": key,
                "dataset_fingerprint": dataset.metadata.fingerprint,
                "asset": dataset.metadata.instrument.asset.value,
                "engines": [item.value for item in engines],
                "duration_count": len(durations),
                "fold_count": len(dataset.folds),
                "planned_evaluation_count": len(engines)
                * len(durations)
                * len(dataset.folds),
                "library_versions": versions,
                "dry_run": False,
            }
        },
    )
    started = perf_counter()
    tracemalloc.start()
    all_results: list[ScreeningResult] = []
    computed = 0
    try:
        for engine in engines:
            for duration in durations:
                combination_started = perf_counter()
                evaluations = evaluate_combination(
                    dataset=dataset,
                    engine_name=engine,
                    duration=duration,
                    configuration=configuration,
                )
                await persist_evaluations(session, run_id, evaluations, timestamp)
                computed += len(evaluations)
                all_results.append(aggregate_evaluations(evaluations))
                logger.info(
                    "ML screening combination completed",
                    extra={
                        "structured_fields": {
                            "run_id": str(run_id),
                            "engine": engine.value,
                            "duration_seconds": int(duration.total_seconds()),
                            "folds_completed": len(evaluations),
                            "elapsed_seconds": perf_counter() - combination_started,
                            "evaluations_computed": computed,
                            "evaluations_reused": 0,
                        }
                    },
                )
        results = apply_pareto(tuple(all_results))
        elapsed = perf_counter() - started
        await complete_run(
            session, run_id, results, computed, elapsed, datetime.now(UTC)
        )
    except Exception as error:
        await fail_run(session, run_id, error, datetime.now(UTC))
        raise
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(
        "ML model screening completed",
        extra={
            "structured_fields": {
                "run_id": str(run_id),
                "status": "completed",
                "evaluations_computed": computed,
                "evaluations_reused": 0,
                "results_generated": len(results),
                "pareto_retained_count": sum(item.pareto_retained for item in results),
                "compute_avoided": 0,
                "total_elapsed_seconds": elapsed,
                "approximate_peak_resident_memory": peak_memory,
            }
        },
    )
    return ScreeningExecutionResult(
        run_id,
        key,
        False,
        computed,
        0,
        len(results),
        sum(item.pareto_retained for item in results),
    )
