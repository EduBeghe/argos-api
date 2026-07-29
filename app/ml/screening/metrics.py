import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    sample_count: int
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_prediction: float
    mean_actual_return: float
    pearson_correlation: float | None
    spearman_correlation: float | None
    directional_accuracy: float
    zero_prediction_mae: float
    mae_improvement_over_zero: float
    positive_prediction_count: int
    negative_prediction_count: int


def _correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    from scipy.stats import rankdata

    return np.asarray(rankdata(values, method="average"), dtype=np.float64)


def calculate_metrics(
    predictions: NDArray[np.float64], actual: NDArray[np.float64]
) -> EvaluationMetrics:
    if predictions.shape != actual.shape or predictions.ndim != 1 or not len(actual):
        raise ValueError(
            "predictions and actual returns must be aligned non-empty vectors"
        )
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(actual)):
        raise ValueError("screening metrics require finite values")
    errors = predictions - actual
    absolute = np.abs(errors)
    mae = float(np.mean(absolute))
    zero_mae = float(np.mean(np.abs(actual)))
    return EvaluationMetrics(
        sample_count=len(actual),
        mean_absolute_error=mae,
        root_mean_squared_error=float(np.sqrt(np.mean(errors * errors))),
        mean_prediction=float(np.mean(predictions)),
        mean_actual_return=float(np.mean(actual)),
        pearson_correlation=_correlation(predictions, actual),
        spearman_correlation=_correlation(_ranks(predictions), _ranks(actual)),
        directional_accuracy=float(np.mean(np.sign(predictions) == np.sign(actual))),
        zero_prediction_mae=zero_mae,
        mae_improvement_over_zero=zero_mae - mae,
        positive_prediction_count=int(np.count_nonzero(predictions > 0)),
        negative_prediction_count=int(np.count_nonzero(predictions < 0)),
    )
