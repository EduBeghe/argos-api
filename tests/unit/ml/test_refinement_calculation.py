import numpy as np
import pytest

from app.ml.refinement.calculation import (
    RefinementFoldResult,
    _aggregate_confidence_buckets,
    _confidence_bucket,
    _fit_predict,
    _signed_classification_scores,
    _training_class_count,
)
from app.ml.refinement.contracts import PredictionObjective
from app.ml.refinement.planning import PlannedEvaluation
from app.ml.screening.config import ScreeningEngine


def candidate(objective: PredictionObjective) -> PlannedEvaluation:
    return PlannedEvaluation(
        "x" * 64,
        ScreeningEngine.RIDGE,
        1800,
        objective,
        (("alpha", 1.0), ("fit_intercept", True)),
        "all",
        None,
        None,
    )


@pytest.mark.parametrize("objective", tuple(PredictionObjective))
def test_ridge_supports_each_prediction_objective(
    objective: PredictionObjective,
) -> None:
    random = np.random.default_rng(1729)
    training_x = random.normal(size=(90, 4)).astype(np.float32)
    training_y = np.resize(np.asarray([-0.02, 0.0, 0.02], dtype=np.float32), 90)
    validation_x = random.normal(size=(12, 4)).astype(np.float32)
    training, validation = _fit_predict(
        candidate(objective),
        training_x,
        training_y,
        validation_x,
        0.003,
        1729,
    )
    assert training.shape == (90,)
    assert validation.shape == (12,)
    assert np.all(np.isfinite(training))
    assert np.all(np.isfinite(validation))


def test_confidence_threshold_uses_training_and_costs_reduce_net_evidence() -> None:
    training = np.asarray([-0.9, -0.2, 0.1, 0.8])
    validation = np.asarray([-0.8, 0.7, 0.1, -0.05])
    actual = np.asarray([-0.02, 0.03, 0.01, -0.01])
    free = _confidence_bucket(
        fraction=0.5,
        training_predictions=training,
        validation_predictions=validation,
        actual=actual,
        round_trip_cost=0.0,
    )
    costly = _confidence_bucket(
        fraction=0.5,
        training_predictions=training,
        validation_predictions=validation,
        actual=actual,
        round_trip_cost=0.01,
    )
    assert (
        free["training_confidence_threshold"] == costly["training_confidence_threshold"]
    )
    costly_return = costly["estimated_net_return"]
    free_return = free["estimated_net_return"]
    assert isinstance(costly_return, float)
    assert isinstance(free_return, float)
    assert costly_return < free_return


def test_signed_binary_confidence_selects_both_directions() -> None:
    probabilities = np.asarray([0.01, 0.50, 0.99])
    signed = 2 * probabilities - 1
    result = _confidence_bucket(
        fraction=2 / 3,
        training_predictions=signed,
        validation_predictions=signed,
        actual=np.asarray([-0.03, 0.0, 0.03]),
        round_trip_cost=0.001,
    )
    assert signed.tolist() == pytest.approx([-0.98, 0.0, 0.98])
    assert result["negative_signal_count"] == 1
    assert result["positive_signal_count"] == 1
    assert result["negative_signals"]["directional_accuracy"] == 1.0  # type: ignore[index]
    assert result["positive_signals"]["directional_accuracy"] == 1.0  # type: ignore[index]


def test_direction_classification_uses_the_positive_class_column() -> None:
    probabilities = np.asarray([[0.75, 0.25], [0.10, 0.90]], dtype=np.float64)

    scores = _signed_classification_scores(
        probabilities,
        np.asarray([0, 1], dtype=np.int64),
        PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION,
    )

    assert scores.tolist() == pytest.approx([-0.5, 0.8])


@pytest.mark.parametrize(
    ("classes", "probabilities", "expected"),
    [
        ([0, 1], [[0.75, 0.25], [0.10, 0.90]], [-0.75, -0.10]),
        ([0, 2], [[0.75, 0.25], [0.10, 0.90]], [-0.50, 0.80]),
        ([1, 2], [[0.75, 0.25], [0.10, 0.90]], [0.25, 0.90]),
    ],
)
def test_cost_adjusted_scores_preserve_partial_class_semantics(
    classes: list[int],
    probabilities: list[list[float]],
    expected: list[float],
) -> None:
    scores = _signed_classification_scores(
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(classes, dtype=np.int64),
        PredictionObjective.COST_ADJUSTED_ACTION_CLASSIFICATION,
    )

    assert scores.tolist() == pytest.approx(expected)


@pytest.mark.parametrize(
    ("actual", "expected_positive", "expected_negative"),
    [
        ([0.03, -0.001], True, False),
        ([0.001, -0.03], False, True),
        ([0.03, -0.03], True, True),
        ([0.001, -0.001], False, False),
    ],
)
def test_direction_specific_bucket_economics(
    actual: list[float], expected_positive: bool, expected_negative: bool
) -> None:
    result = _confidence_bucket(
        fraction=1.0,
        training_predictions=np.asarray([0.9, -0.9]),
        validation_predictions=np.asarray([0.9, -0.9]),
        actual=np.asarray(actual),
        round_trip_cost=0.01,
    )
    positive = result["positive_signals"]
    negative = result["negative_signals"]
    assert isinstance(positive, dict)
    assert isinstance(negative, dict)
    assert (positive["estimated_net_return"] > 0) is expected_positive
    assert (negative["estimated_net_return"] > 0) is expected_negative


def test_missing_training_classes_are_detected_for_both_classifiers() -> None:
    values = np.asarray([0.001, 0.001], dtype=np.float32)
    assert (
        _training_class_count(
            PredictionObjective.FUTURE_DIRECTION_CLASSIFICATION, values, 0.01
        )
        == 1
    )
    assert (
        _training_class_count(
            PredictionObjective.COST_ADJUSTED_ACTION_CLASSIFICATION, values, 0.01
        )
        == 1
    )


def test_confidence_evidence_is_aggregated_within_the_same_bucket() -> None:
    first = _confidence_bucket(
        fraction=0.1,
        training_predictions=np.asarray([0.9, -0.9]),
        validation_predictions=np.asarray([0.9, -0.9]),
        actual=np.asarray([0.03, -0.03]),
        round_trip_cost=0.01,
    )
    second = _confidence_bucket(
        fraction=0.1,
        training_predictions=np.asarray([0.9, -0.9]),
        validation_predictions=np.asarray([0.9, -0.9]),
        actual=np.asarray([-0.01, 0.01]),
        round_trip_cost=0.01,
    )
    folds = [
        RefinementFoldResult("a", 10, 2, (), {}, (first,), 0.01),
        RefinementFoldResult("b", 10, 2, (), {}, (second,), 0.01),
    ]
    aggregate = _aggregate_confidence_buckets(folds)[0]
    assert aggregate["observation_count"] == 4
    assert aggregate["folds_with_positive_net_return"] == 1
    assert aggregate["folds_with_favorable_direction"] == 1
    assert aggregate["median_net_return"] == pytest.approx(0.0)
    assert aggregate["worst_net_return"] == pytest.approx(-0.02)
