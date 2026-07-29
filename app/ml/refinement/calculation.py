import math
from dataclasses import asdict, dataclass
from datetime import timedelta
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from app.ml.contract import FINAL_TEST_START
from app.ml.dataset import MLExperimentDataset
from app.ml.gaps import datetime_to_ns
from app.ml.registry import VARIABLE_REGISTRY
from app.ml.screening.metrics import calculate_metrics

from .contracts import (
    MLRefinementRequest,
    PredictionObjective,
    RefinementClassification,
)
from .evidence import RefinementEvidence, classify_refinement_evidence
from .planning import PlannedEvaluation


@dataclass(frozen=True, slots=True)
class RefinementFoldResult:
    fold_name: str
    training_row_count: int
    validation_row_count: int
    selected_variables: tuple[str, ...]
    metrics: dict[str, object]
    confidence_metrics: tuple[dict[str, object], ...]
    economic_threshold: float


@dataclass(frozen=True, slots=True)
class RefinementEvaluationResult:
    candidate: PlannedEvaluation
    folds: tuple[RefinementFoldResult, ...]
    aggregate_metrics: dict[str, object]
    classification: RefinementClassification
    failed_rules: tuple[str, ...]
    elapsed_seconds: float


def _selected_columns(
    candidate: PlannedEvaluation,
    training_features: NDArray[np.float32],
    training_target: NDArray[np.float32],
) -> NDArray[np.int64]:
    names = tuple(item.name for item in VARIABLE_REGISTRY)
    if candidate.feature_strategy == "all":
        return np.arange(len(names), dtype=np.int64)
    if candidate.feature_strategy == "explicit":
        assert candidate.explicit_variables is not None
        positions = {name: index for index, name in enumerate(names)}
        return np.asarray(
            [positions[name] for name in candidate.explicit_variables], dtype=np.int64
        )
    assert candidate.top_k is not None
    centered_target = training_target.astype(np.float64) - float(
        np.mean(training_target)
    )
    centered_features = training_features.astype(np.float64) - np.mean(
        training_features, axis=0
    )
    scores = np.abs(centered_features.T @ centered_target)
    scores[~np.isfinite(scores)] = -math.inf
    order = np.lexsort((np.arange(len(names)), -scores))
    return np.sort(order[: candidate.top_k].astype(np.int64))


def _fit_predict(
    candidate: PlannedEvaluation,
    training_x: NDArray[np.float32],
    training_y: NDArray[np.float32],
    validation_x: NDArray[np.float32],
    economic_threshold: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    parameters = dict(candidate.hyperparameters)
    objective = candidate.objective
    regression = objective is PredictionObjective.FUTURE_RETURN_REGRESSION
    labels: NDArray[np.float32] | NDArray[np.int64]
    if objective is PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION:
        labels = (training_y > 0).astype(np.int64)
    elif objective is PredictionObjective.COST_ADJUSTED_ACTION_CLASSIFICATION:
        labels = np.where(
            training_y > economic_threshold,
            2,
            np.where(training_y < -economic_threshold, 0, 1),
        )
    else:
        labels = training_y
    if candidate.engine.value == "ridge":
        if regression:
            from sklearn.linear_model import Ridge

            model = Ridge(**parameters, random_state=None)
        else:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                C=1.0 / max(float(parameters["alpha"]), 1e-12),
                fit_intercept=bool(parameters["fit_intercept"]),
                random_state=seed,
                max_iter=500,
            )
    elif candidate.engine.value == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        model_type = LGBMRegressor if regression else LGBMClassifier
        model = model_type(
            **parameters,  # type: ignore[arg-type]
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=1,
        )
    else:
        from catboost import CatBoostClassifier, CatBoostRegressor

        model_type = CatBoostRegressor if regression else CatBoostClassifier
        model = model_type(
            **parameters,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
            thread_count=1,
        )
    model.fit(training_x, labels)
    if regression:
        return (
            np.asarray(model.predict(training_x), dtype=np.float64),
            np.asarray(model.predict(validation_x), dtype=np.float64),
        )
    training_probability = np.asarray(model.predict_proba(training_x), dtype=np.float64)
    validation_probability = np.asarray(
        model.predict_proba(validation_x), dtype=np.float64
    )
    if training_probability.shape[1] == 2:
        return training_probability[:, 1], validation_probability[:, 1]
    classes = np.asarray(model.classes_)
    signed = np.where(classes == 2, 1.0, np.where(classes == 0, -1.0, 0.0))
    return training_probability @ signed, validation_probability @ signed


def _classification_metrics(
    predictions: NDArray[np.float64],
    actual: NDArray[np.float64],
    threshold: float,
    *,
    three_class: bool,
) -> dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    if three_class:
        actual_labels = np.where(
            actual > threshold, 2, np.where(actual < -threshold, 0, 1)
        )
        predicted_labels = np.where(
            predictions > 0.5, 2, np.where(predictions < -0.5, 0, 1)
        )
        average = "macro"
    else:
        actual_labels = (actual > threshold).astype(np.int64)
        predicted_labels = (predictions > 0.5).astype(np.int64)
        average = "binary"
    unique = np.unique(actual_labels)
    return {
        "accuracy": float(accuracy_score(actual_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual_labels, predicted_labels)
        ),
        "precision": float(
            precision_score(
                actual_labels, predicted_labels, average=average, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                actual_labels, predicted_labels, average=average, zero_division=0
            )
        ),
        "roc_auc": float(roc_auc_score(actual_labels, predictions))
        if not three_class and len(unique) == 2
        else None,
        "down_examples": int(np.count_nonzero(actual_labels == 0))
        if three_class
        else None,
        "no_trade_examples": int(np.count_nonzero(actual_labels == 1))
        if three_class
        else None,
        "up_examples": int(np.count_nonzero(actual_labels == 2))
        if three_class
        else None,
        "negative_examples": int(np.count_nonzero(actual_labels == 0))
        if not three_class
        else None,
        "positive_examples": int(np.count_nonzero(actual_labels == 1))
        if not three_class
        else None,
    }


def _confidence_bucket(
    *,
    fraction: float,
    training_predictions: NDArray[np.float64],
    validation_predictions: NDArray[np.float64],
    actual: NDArray[np.float64],
    round_trip_cost: float,
) -> dict[str, object]:
    confidence = np.abs(training_predictions)
    cutoff = float(np.quantile(confidence, 1.0 - fraction))
    selected = np.abs(validation_predictions) >= cutoff
    returns = actual[selected]
    signs = np.sign(validation_predictions[selected])
    aligned = returns * signs
    net = aligned - round_trip_cost
    return {
        "fraction": fraction,
        "training_confidence_threshold": cutoff,
        "observation_count": int(np.count_nonzero(selected)),
        "coverage": float(np.mean(selected)),
        "directional_accuracy": float(np.mean(aligned > 0)) if len(aligned) else None,
        "mean_future_return": float(np.mean(returns)) if len(returns) else None,
        "median_future_return": float(np.median(returns)) if len(returns) else None,
        "estimated_net_return": float(np.mean(net)) if len(net) else None,
        "cost_covering_fraction": float(np.mean(aligned > round_trip_cost))
        if len(aligned)
        else None,
        "positive_signal_count": int(np.count_nonzero(signs > 0)),
        "negative_signal_count": int(np.count_nonzero(signs < 0)),
    }


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError("expected numeric metric")
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("expected integer metric")
    return value


def evaluate_refinement_candidate(
    *,
    dataset: MLExperimentDataset,
    candidate: PlannedEvaluation,
    request: MLRefinementRequest,
) -> RefinementEvaluationResult:
    if np.any(dataset.boundary_time_ns >= datetime_to_ns(FINAL_TEST_START)):
        raise ValueError("refinement dataset contains protected final-test evidence")
    started = perf_counter()
    duration = timedelta(seconds=candidate.future_duration_seconds)
    target = dataset.targets.for_duration(duration)
    round_trip_cost = float(2 * (request.fee_rate + request.slippage_rate))
    folds: list[RefinementFoldResult] = []
    for fold in dataset.folds:
        eligible = dataset.feature_eligible & target.eligible
        training = fold.training_indices[eligible[fold.training_indices]]
        validation = fold.validation_indices[eligible[fold.validation_indices]]
        if not len(training) or not len(validation):
            continue
        columns = _selected_columns(
            candidate, dataset.features[training], target.values[training]
        )
        training_x = dataset.features[np.ix_(training, columns)]
        validation_x = dataset.features[np.ix_(validation, columns)]
        training_predictions, validation_predictions = _fit_predict(
            candidate,
            training_x,
            target.values[training],
            validation_x,
            round_trip_cost,
            request.deterministic_seed,
        )
        actual = target.values[validation].astype(np.float64)
        metrics = (
            asdict(calculate_metrics(validation_predictions, actual))
            if candidate.objective is PredictionObjective.FUTURE_RETURN_REGRESSION
            else _classification_metrics(
                validation_predictions,
                actual,
                round_trip_cost
                if candidate.objective
                is PredictionObjective.COST_ADJUSTED_ACTION_CLASSIFICATION
                else 0.0,
                three_class=(
                    candidate.objective
                    is PredictionObjective.COST_ADJUSTED_ACTION_CLASSIFICATION
                ),
            )
        )
        buckets = tuple(
            _confidence_bucket(
                fraction=float(fraction),
                training_predictions=training_predictions,
                validation_predictions=validation_predictions,
                actual=actual,
                round_trip_cost=round_trip_cost,
            )
            for fraction in sorted(request.confidence_buckets)
        )
        folds.append(
            RefinementFoldResult(
                fold.definition.name,
                len(training),
                len(validation),
                tuple(VARIABLE_REGISTRY[index].name for index in columns),
                metrics,
                buckets,
                round_trip_cost,
            )
        )
    best = [
        bucket
        for fold in folds
        for bucket in fold.confidence_metrics
        if bucket["estimated_net_return"] is not None
    ]
    evidence = RefinementEvidence(
        fold_count=len(folds),
        folds_beating_benchmark=sum(
            _number(fold.metrics.get("mae_improvement_over_zero", 0)) > 0
            or _number(fold.metrics.get("balanced_accuracy", 0)) > 0.5
            for fold in folds
        ),
        folds_with_positive_direction=sum(
            any(
                bucket["directional_accuracy"] is not None
                and _number(bucket["directional_accuracy"]) > 0.5
                for bucket in fold.confidence_metrics
            )
            for fold in folds
        ),
        best_confidence_net_return=max(
            (_number(item["estimated_net_return"]) for item in best), default=None
        ),
        best_confidence_observations=max(
            (_integer(item["observation_count"]) for item in best), default=0
        ),
    )
    classification, failed = classify_refinement_evidence(evidence)
    aggregate: dict[str, object] = {
        "fold_count": len(folds),
        "folds_beating_benchmark": evidence.folds_beating_benchmark,
        "folds_with_positive_direction": evidence.folds_with_positive_direction,
        "best_confidence_net_return": evidence.best_confidence_net_return,
        "best_confidence_observations": evidence.best_confidence_observations,
    }
    return RefinementEvaluationResult(
        candidate,
        tuple(folds),
        aggregate,
        classification,
        failed,
        perf_counter() - started,
    )
