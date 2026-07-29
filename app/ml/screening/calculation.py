import gc
import math
from dataclasses import dataclass, replace
from datetime import timedelta
from statistics import median
from time import perf_counter

import numpy as np

from app.ml.dataset import MLExperimentDataset
from app.ml.errors import MLDatasetError

from .config import ModelScreeningConfiguration, ScreeningEngine
from .engines import create_engine
from .metrics import EvaluationMetrics, calculate_metrics


@dataclass(frozen=True, slots=True)
class FoldEvaluation:
    engine: ScreeningEngine
    future_duration_seconds: int
    fold_name: str
    training_row_count: int
    validation_row_count: int
    metrics: EvaluationMetrics
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    engine: ScreeningEngine
    future_duration_seconds: int
    folds_evaluated: int
    total_validation_samples: int
    median_mae_improvement_over_zero: float
    worst_fold_mae_improvement_over_zero: float
    median_directional_accuracy: float
    worst_fold_directional_accuracy: float
    median_pearson_correlation: float | None
    median_spearman_correlation: float | None
    folds_beating_zero: int
    folds_with_positive_spearman: int
    total_training_time_seconds: float
    pareto_retained: bool = False
    display_rank: int = 0


def assert_final_test_isolation(
    dataset: MLExperimentDataset, training: np.ndarray, validation: np.ndarray
) -> None:
    final = dataset.final_test_indices
    if np.intersect1d(training, validation).size:
        raise MLDatasetError("training and validation indices overlap")
    if np.intersect1d(training, final).size or np.intersect1d(validation, final).size:
        raise MLDatasetError("protected final-test indices entered model screening")


def evaluate_combination(
    *,
    dataset: MLExperimentDataset,
    engine_name: ScreeningEngine,
    duration: timedelta,
    configuration: ModelScreeningConfiguration,
) -> tuple[FoldEvaluation, ...]:
    target = dataset.targets.for_duration(duration)
    evaluations: list[FoldEvaluation] = []
    for fold in dataset.folds:
        indices = dataset.eligible_indices(
            fold_name=fold.definition.name, duration=duration
        )
        assert_final_test_isolation(
            dataset, indices.training_indices, indices.validation_indices
        )
        if not len(indices.training_indices) or not len(indices.validation_indices):
            continue
        started = perf_counter()
        engine = create_engine(engine_name, configuration)
        predictions = engine.fit_predict(
            dataset.features[indices.training_indices],
            target.values[indices.training_indices],
            dataset.features[indices.validation_indices],
        )
        actual = target.values[indices.validation_indices].astype(np.float64)
        if predictions.shape != actual.shape or not np.all(np.isfinite(predictions)):
            raise ValueError("engine returned non-finite or misaligned predictions")
        evaluations.append(
            FoldEvaluation(
                engine_name,
                int(duration.total_seconds()),
                fold.definition.name,
                len(indices.training_indices),
                len(indices.validation_indices),
                calculate_metrics(predictions, actual),
                perf_counter() - started,
            )
        )
        del engine, predictions, actual
        gc.collect()
    return tuple(evaluations)


def _nullable_median(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return median(finite) if finite else None


def aggregate_evaluations(evaluations: tuple[FoldEvaluation, ...]) -> ScreeningResult:
    if not evaluations:
        raise ValueError("at least one fold evaluation is required")
    first = evaluations[0]
    improvements = [item.metrics.mae_improvement_over_zero for item in evaluations]
    accuracies = [item.metrics.directional_accuracy for item in evaluations]
    return ScreeningResult(
        first.engine,
        first.future_duration_seconds,
        len(evaluations),
        sum(item.validation_row_count for item in evaluations),
        median(improvements),
        min(improvements),
        median(accuracies),
        min(accuracies),
        _nullable_median([item.metrics.pearson_correlation for item in evaluations]),
        _nullable_median([item.metrics.spearman_correlation for item in evaluations]),
        sum(value > 0 for value in improvements),
        sum(
            item.metrics.spearman_correlation is not None
            and item.metrics.spearman_correlation > 0
            for item in evaluations
        ),
        sum(item.elapsed_seconds for item in evaluations),
    )


def _predictive_dimensions(result: ScreeningResult) -> tuple[float, float, float, int]:
    return (
        result.median_mae_improvement_over_zero,
        result.worst_fold_mae_improvement_over_zero,
        result.median_spearman_correlation
        if result.median_spearman_correlation is not None
        else -math.inf,
        result.folds_beating_zero,
    )


def apply_pareto(results: tuple[ScreeningResult, ...]) -> tuple[ScreeningResult, ...]:
    retained: list[ScreeningResult] = []
    for candidate in results:
        dimensions = _predictive_dimensions(candidate)
        dominated = any(
            other is not candidate
            and all(
                a >= b
                for a, b in zip(_predictive_dimensions(other), dimensions, strict=True)
            )
            and any(
                a > b
                for a, b in zip(_predictive_dimensions(other), dimensions, strict=True)
            )
            for other in results
        )
        retained.append(replace(candidate, pareto_retained=not dominated))
    ordered = sorted(
        retained,
        key=lambda item: (
            not item.pareto_retained,
            tuple(-value for value in _predictive_dimensions(item)),
            item.total_training_time_seconds,
            item.engine.value,
            item.future_duration_seconds,
        ),
    )
    return tuple(
        replace(item, display_rank=index + 1) for index, item in enumerate(ordered)
    )
