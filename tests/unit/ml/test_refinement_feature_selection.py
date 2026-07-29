import numpy as np

from app.ml.refinement.calculation import _selected_columns
from app.ml.refinement.contracts import PredictionObjective
from app.ml.refinement.planning import PlannedEvaluation
from app.ml.screening.config import ScreeningEngine


def candidate(*, strategy: str, top_k: int | None = None) -> PlannedEvaluation:
    return PlannedEvaluation(
        "f" * 64,
        ScreeningEngine.RIDGE,
        1800,
        PredictionObjective.FUTURE_RETURN_REGRESSION,
        (("alpha", 1.0), ("fit_intercept", True)),
        strategy,
        None,
        top_k,
    )


def test_top_k_uses_training_only_and_is_deterministic() -> None:
    rows = 20
    columns = 126
    training = np.zeros((rows, columns), dtype=np.float32)
    target = np.linspace(-1, 1, rows, dtype=np.float32)
    training[:, 7] = target
    training[:, 11] = target * 0.5
    first = _selected_columns(candidate(strategy="top_k", top_k=2), training, target)
    unrelated_validation = np.random.default_rng(10).normal(size=(100, columns))
    del unrelated_validation
    second = _selected_columns(candidate(strategy="top_k", top_k=2), training, target)
    assert first.tolist() == [7, 11]
    assert np.array_equal(first, second)


def test_all_selection_returns_registry_width() -> None:
    training = np.zeros((2, 126), dtype=np.float32)
    target = np.zeros(2, dtype=np.float32)
    assert len(_selected_columns(candidate(strategy="all"), training, target)) == 126
