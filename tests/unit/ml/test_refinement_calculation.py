import numpy as np
import pytest

from app.ml.refinement.calculation import _confidence_bucket, _fit_predict
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
