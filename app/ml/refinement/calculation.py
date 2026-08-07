import math
from dataclasses import asdict, dataclass
from datetime import timedelta
from decimal import Decimal
from time import perf_counter
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from app.ml.campaign.economic_v2_calculation import (
    aggregate_policy_folds,
    evaluate_policy_grid_fold,
)
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
    incomplete_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RefinementEvaluationResult:
    candidate: PlannedEvaluation
    folds: tuple[RefinementFoldResult, ...]
    aggregate_metrics: dict[str, object]
    classification: RefinementClassification
    failed_rules: tuple[str, ...]
    elapsed_seconds: float


class FittedRefinementModel(Protocol):
    def fit(self, features: object, labels: object) -> object: ...
    def predict(self, features: object) -> object: ...
    def predict_proba(self, features: object) -> object: ...

    classes_: object


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
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64] | None,
]:
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
    model = fit_refinement_model(
        candidate=candidate, training_x=training_x, labels=labels, seed=seed
    )
    if regression:
        return (
            np.asarray(model.predict(training_x), dtype=np.float64),
            np.asarray(model.predict(validation_x), dtype=np.float64),
            None,
        )
    training_probability = np.asarray(model.predict_proba(training_x), dtype=np.float64)
    validation_probability = np.asarray(
        model.predict_proba(validation_x), dtype=np.float64
    )
    classes = np.asarray(model.classes_, dtype=np.int64)
    return (
        _signed_classification_scores(training_probability, classes, objective),
        _signed_classification_scores(validation_probability, classes, objective),
        _predicted_class_labels(validation_probability, classes),
    )


def fit_refinement_model(
    *,
    candidate: PlannedEvaluation,
    training_x: NDArray[np.float32],
    labels: NDArray[np.float32] | NDArray[np.int64],
    seed: int,
) -> FittedRefinementModel:
    """Fit one frozen engine with the same deterministic production settings."""
    parameters = dict(candidate.hyperparameters)
    regression = candidate.objective is PredictionObjective.FUTURE_RETURN_REGRESSION
    model: object
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
    fitted = cast(FittedRefinementModel, model)
    fitted.fit(training_x, labels)
    return fitted


def _signed_classification_scores(
    probabilities: NDArray[np.float64],
    classes: NDArray[np.int64],
    objective: PredictionObjective,
) -> NDArray[np.float64]:
    if objective is PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION:
        positive_class = np.flatnonzero(classes == 1)
        if len(positive_class) != 1:
            raise ValueError("direction classifier must expose the positive class")
        return 2 * probabilities[:, int(positive_class[0])] - 1

    if objective is not PredictionObjective.COST_ADJUSTED_ACTION_CLASSIFICATION:
        raise ValueError("classification scores require a classification objective")
    if np.any(~np.isin(classes, (0, 1, 2))):
        raise ValueError("cost-adjusted classifier exposed an unknown class")
    signed_classes = np.where(classes == 2, 1.0, np.where(classes == 0, -1.0, 0.0))
    return probabilities @ signed_classes


def _predicted_class_labels(
    probabilities: NDArray[np.float64], classes: NDArray[np.int64]
) -> NDArray[np.int64]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("class probabilities must align with model classes")
    return classes[np.argmax(probabilities, axis=1)]


def _classification_metrics(
    signed_scores: NDArray[np.float64],
    predicted_labels: NDArray[np.int64],
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
        average = "macro"
    else:
        actual_labels = (actual > threshold).astype(np.int64)
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
        "roc_auc": float(roc_auc_score(actual_labels, (signed_scores + 1) / 2))
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

    def direction_metrics(mask: NDArray[np.bool_]) -> dict[str, object]:
        direction_returns = returns[mask]
        direction_aligned = aligned[mask]
        direction_net = net[mask]
        return {
            "observation_count": int(np.count_nonzero(mask)),
            "directional_accuracy": (
                float(np.mean(direction_aligned > 0))
                if len(direction_aligned)
                else None
            ),
            "mean_future_return": (
                float(np.mean(direction_returns)) if len(direction_returns) else None
            ),
            "median_future_return": (
                float(np.median(direction_returns)) if len(direction_returns) else None
            ),
            "estimated_net_return": (
                float(np.mean(direction_net)) if len(direction_net) else None
            ),
            "cost_covering_fraction": (
                float(np.mean(direction_aligned > round_trip_cost))
                if len(direction_aligned)
                else None
            ),
        }

    positive = signs > 0
    negative = signs < 0
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
        "positive_signals": direction_metrics(positive),
        "negative_signals": direction_metrics(negative),
    }


def _aggregate_confidence_buckets(
    folds: list[RefinementFoldResult],
) -> tuple[dict[str, object], ...]:
    fractions = sorted(
        {
            _number(bucket["fraction"])
            for fold in folds
            for bucket in fold.confidence_metrics
        }
    )
    aggregates: list[dict[str, object]] = []
    for fraction in fractions:
        buckets = [
            bucket
            for fold in folds
            for bucket in fold.confidence_metrics
            if _number(bucket["fraction"]) == fraction
        ]
        usable = [
            bucket for bucket in buckets if bucket["estimated_net_return"] is not None
        ]
        observations = sum(_integer(bucket["observation_count"]) for bucket in buckets)
        net_returns = [_number(bucket["estimated_net_return"]) for bucket in usable]
        weighted_net = sum(
            _number(bucket["estimated_net_return"])
            * _integer(bucket["observation_count"])
            for bucket in usable
        )
        coverage_weight = sum(_number(bucket["coverage"]) for bucket in buckets)
        aggregates.append(
            {
                "fraction": fraction,
                "observation_count": observations,
                "folds_with_evidence": len(usable),
                "folds_with_positive_net_return": sum(
                    value > 0 for value in net_returns
                ),
                "folds_with_favorable_direction": sum(
                    bucket["directional_accuracy"] is not None
                    and _number(bucket["directional_accuracy"]) > 0.5
                    for bucket in buckets
                ),
                "median_net_return": (
                    float(np.median(net_returns)) if net_returns else None
                ),
                "worst_net_return": min(net_returns) if net_returns else None,
                "observation_weighted_net_return": (
                    weighted_net / observations if observations else None
                ),
                "aggregate_coverage": (
                    coverage_weight / len(buckets) if buckets else 0.0
                ),
            }
        )
    return tuple(aggregates)


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError("expected numeric metric")
    return float(value)


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise ValueError("expected integer metric")
    return value


def _training_class_count(
    objective: PredictionObjective,
    target: NDArray[np.float32],
    economic_threshold: float,
) -> int | None:
    if objective is PredictionObjective.FUTURE_RETURN_REGRESSION:
        return None
    if objective is PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION:
        return len(np.unique(target > 0))
    labels = np.where(
        target > economic_threshold,
        2,
        np.where(target < -economic_threshold, 0, 1),
    )
    return len(np.unique(labels))


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
    policy_folds: list[tuple[dict[str, object], ...]] = []
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
        class_count = _training_class_count(
            candidate.objective, target.values[training], round_trip_cost
        )
        validation_class_count = _training_class_count(
            candidate.objective, target.values[validation], round_trip_cost
        )
        if (
            class_count is not None
            and validation_class_count is not None
            and (class_count < 2 or validation_class_count < 2)
        ):
            folds.append(
                RefinementFoldResult(
                    fold.definition.name,
                    len(training),
                    len(validation),
                    tuple(VARIABLE_REGISTRY[index].name for index in columns),
                    {},
                    (),
                    round_trip_cost,
                    "classification_fold_contains_single_class",
                )
            )
            continue
        training_predictions, validation_predictions, predicted_labels = _fit_predict(
            candidate,
            training_x,
            target.values[training],
            validation_x,
            round_trip_cost,
            request.deterministic_seed,
        )
        actual = target.values[validation].astype(np.float64)
        if candidate.objective is PredictionObjective.FUTURE_RETURN_REGRESSION:
            metrics = asdict(calculate_metrics(validation_predictions, actual))
        else:
            if predicted_labels is None:
                raise ValueError("classification evaluation requires predicted labels")
            metrics = _classification_metrics(
                validation_predictions,
                predicted_labels,
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
        if request.economic_policy_search is not None:
            policy_folds.append(
                evaluate_policy_grid_fold(
                    dataset=dataset,
                    training=training,
                    validation=validation,
                    training_scores=training_predictions,
                    validation_scores=validation_predictions,
                    horizon_seconds=candidate.future_duration_seconds,
                    fee_rate=request.fee_rate,
                    slippage_rate=request.slippage_rate,
                )
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
    confidence_aggregates = _aggregate_confidence_buckets(folds)
    selected = max(
        confidence_aggregates,
        key=lambda item: (
            _integer(item["folds_with_positive_net_return"]),
            _integer(item["folds_with_favorable_direction"]),
            _number(item["median_net_return"])
            if item["median_net_return"] is not None
            else -math.inf,
            _integer(item["observation_count"]),
            -_number(item["fraction"]),
        ),
        default=None,
    )
    evidence = RefinementEvidence(
        fold_count=len(folds),
        complete_fold_count=sum(fold.incomplete_reason is None for fold in folds),
        folds_beating_benchmark=sum(
            _number(fold.metrics.get("mae_improvement_over_zero", 0)) > 0
            or _number(fold.metrics.get("balanced_accuracy", 0)) > 0.5
            for fold in folds
            if fold.incomplete_reason is None
        ),
        selected_bucket_fraction=(
            _number(selected["fraction"]) if selected is not None else None
        ),
        selected_bucket_observations=(
            _integer(selected["observation_count"]) if selected is not None else 0
        ),
        selected_bucket_positive_net_folds=(
            _integer(selected["folds_with_positive_net_return"])
            if selected is not None
            else 0
        ),
        selected_bucket_favorable_direction_folds=(
            _integer(selected["folds_with_favorable_direction"])
            if selected is not None
            else 0
        ),
        selected_bucket_median_net_return=(
            _number(selected["median_net_return"])
            if selected is not None and selected["median_net_return"] is not None
            else None
        ),
        selected_bucket_worst_net_return=(
            _number(selected["worst_net_return"])
            if selected is not None and selected["worst_net_return"] is not None
            else None
        ),
    )
    classification, failed = classify_refinement_evidence(evidence)
    aggregate: dict[str, object] = {
        "fold_count": len(folds),
        "folds_beating_benchmark": evidence.folds_beating_benchmark,
        "complete_fold_count": evidence.complete_fold_count,
        "confidence_bucket_aggregates": confidence_aggregates,
        "selected_confidence_bucket": selected,
    }
    if request.economic_policy_search is not None:
        policy_aggregates = aggregate_policy_folds(tuple(policy_folds))
        selected_policy = min(
            policy_aggregates,
            key=lambda item: (
                -int(str(item["positive_net_folds"])),
                int(str(item["inactive_folds"])),
                -Decimal(str(item["compounded_return_2x"])),
                -Decimal(str(item["worst_net_return"])),
                -Decimal(str(item["median_net_return"])),
                -int(str(item["round_trips"])),
                Decimal(str(item["maximum_drawdown"])),
                str(item["policy_fingerprint"]),
            ),
            default=None,
        )
        aggregate["economic_policy_v2_aggregates"] = policy_aggregates
        aggregate["selected_economic_policy_v2"] = selected_policy
    return RefinementEvaluationResult(
        candidate,
        tuple(folds),
        aggregate,
        classification,
        failed,
        perf_counter() - started,
    )
