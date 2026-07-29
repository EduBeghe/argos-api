from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from app.market.catalog import MARKET_INSTRUMENTS
from app.ml.contract import (
    FUTURE_DURATIONS,
    TemporalFoldDefinition,
    TrainingHistoryMode,
)
from app.ml.dataset import MLExperimentDataset
from app.ml.errors import MLDatasetError
from app.ml.screening.calculation import (
    ScreeningResult,
    _bounded_fold_indices,
    apply_pareto,
    assert_final_test_isolation,
    evaluate_combination,
)
from app.ml.screening.config import (
    ALL_ENGINES,
    DEFAULT_SCREENING_CONFIGURATION,
    ScreeningEngine,
)
from app.ml.screening.engines import create_engine, library_versions
from app.ml.screening.metrics import calculate_metrics
from app.ml.screening.persistence import experiment_key


def test_all_engine_dependencies_import_and_predict_deterministically() -> None:
    versions = library_versions()
    assert set(versions) == {"scikit-learn", "lightgbm", "catboost"}
    generator = np.random.default_rng(42)
    features = generator.normal(size=(60, 6)).astype(np.float32)
    target = (features[:, 0] * 0.2 - features[:, 1] * 0.1).astype(np.float32)
    for name in ALL_ENGINES:
        first = create_engine(name, DEFAULT_SCREENING_CONFIGURATION).fit_predict(
            features[:45], target[:45], features[45:]
        )
        second = create_engine(name, DEFAULT_SCREENING_CONFIGURATION).fit_predict(
            features[:45], target[:45], features[45:]
        )
        assert first.shape == (15,)
        assert np.all(np.isfinite(first))
        np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-10)


class _SyntheticTargets:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def for_duration(
        self, duration: object, *, candidate_indices: np.ndarray | None = None
    ) -> SimpleNamespace:
        del duration
        eligible = np.zeros(len(self.values), dtype=np.bool_)
        if candidate_indices is not None:
            eligible[candidate_indices] = True
        return SimpleNamespace(values=self.values, eligible=eligible)


class _SyntheticDataset:
    def __init__(self) -> None:
        generator = np.random.default_rng(7)
        self.features = generator.normal(size=(80, 126)).astype(np.float32)
        self.targets = _SyntheticTargets(
            (self.features[:, 0] * 0.02 - self.features[:, 1] * 0.01).astype(np.float32)
        )
        origin = datetime(2021, 1, 1, tzinfo=UTC)
        self.boundary_time_ns = np.asarray(
            [
                int((origin + timedelta(minutes=30 * index)).timestamp() * 1e9)
                for index in range(80)
            ],
            dtype=np.int64,
        )
        self.feature_eligible = np.ones(80, dtype=np.bool_)
        training_end = origin + timedelta(minutes=30 * 35)
        self.folds = tuple(
            SimpleNamespace(
                definition=TemporalFoldDefinition(
                    name=f"fold-{index}",
                    history_mode=TrainingHistoryMode.ACCUMULATING,
                    training_start=origin,
                    training_end=training_end,
                    validation_start=origin + timedelta(minutes=30 * (35 + index * 5)),
                    validation_end=origin + timedelta(minutes=30 * (40 + index * 5)),
                ),
                training_indices=np.arange(0, 35, dtype=np.int64),
                validation_indices=np.arange(
                    35 + index * 5, 40 + index * 5, dtype=np.int64
                ),
            )
            for index in range(7)
        )
        self.final_test_indices = np.asarray([78, 79], dtype=np.int64)


def test_real_screening_covers_three_engines_two_durations_and_seven_folds() -> None:
    dataset = cast(MLExperimentDataset, _SyntheticDataset())
    for engine in ALL_ENGINES:
        for duration in FUTURE_DURATIONS[:2]:
            evaluations = evaluate_combination(
                dataset=dataset,
                engine_name=engine,
                duration=duration,
                configuration=DEFAULT_SCREENING_CONFIGURATION,
            )
            assert len(evaluations) == 7
            assert {item.fold_name for item in evaluations} == {
                f"fold-{index}" for index in range(7)
            }


def test_metrics_preserve_undefined_correlations_and_zero_benchmark() -> None:
    actual = np.asarray([-0.2, 0.1, 0.3], dtype=np.float64)
    predictions = np.zeros(3, dtype=np.float64)
    metrics = calculate_metrics(predictions, actual)
    assert metrics.pearson_correlation is None
    assert metrics.spearman_correlation is None
    assert metrics.zero_prediction_mae == pytest.approx(0.2)
    assert metrics.mae_improvement_over_zero == pytest.approx(0.0)


def _result(
    *,
    engine: str,
    median_improvement: float,
    worst: float,
    spearman: float,
    folds: int,
    elapsed: float,
) -> ScreeningResult:
    from app.ml.screening.config import ScreeningEngine

    return ScreeningResult(
        ScreeningEngine(engine),
        1800,
        7,
        100,
        median_improvement,
        worst,
        0.6,
        0.5,
        0.1,
        spearman,
        folds,
        7,
        elapsed,
    )


def test_pareto_retains_non_dominated_and_excludes_dominated() -> None:
    strong = _result(
        engine="ridge",
        median_improvement=0.2,
        worst=0.1,
        spearman=0.3,
        folds=7,
        elapsed=2,
    )
    tradeoff = _result(
        engine="lightgbm",
        median_improvement=0.3,
        worst=0.05,
        spearman=0.4,
        folds=6,
        elapsed=3,
    )
    dominated = _result(
        engine="catboost",
        median_improvement=0.1,
        worst=0.0,
        spearman=0.2,
        folds=5,
        elapsed=10,
    )
    output = apply_pareto((strong, tradeoff, dominated))
    flags = {item.engine.value: item.pareto_retained for item in output}
    assert flags == {"ridge": True, "lightgbm": True, "catboost": False}
    assert [item.display_rank for item in output] == [1, 2, 3]


def test_final_test_indices_fail_closed() -> None:
    dataset = _SyntheticDataset()
    definition = dataset.folds[0].definition
    assert_final_test_isolation(
        cast(MLExperimentDataset, dataset),
        np.asarray([1, 2]),
        np.asarray([36, 37]),
        definition=definition,
        duration=timedelta(minutes=30),
    )
    with pytest.raises(MLDatasetError, match="final-test"):
        assert_final_test_isolation(
            cast(MLExperimentDataset, dataset),
            np.asarray([1, 78]),
            np.asarray([36]),
            definition=definition,
            duration=timedelta(minutes=30),
        )

    with pytest.raises(MLDatasetError, match="validation target crosses"):
        assert_final_test_isolation(
            cast(MLExperimentDataset, dataset),
            np.asarray([], dtype=np.int64),
            np.asarray([39]),
            definition=definition,
            duration=timedelta(days=7),
        )


@pytest.mark.parametrize("duration", [timedelta(minutes=30), timedelta(days=7)])
def test_targets_crossing_segment_end_are_excluded(duration: timedelta) -> None:
    dataset = _SyntheticDataset()
    fold = dataset.folds[0]
    fold.definition = TemporalFoldDefinition(
        name=fold.definition.name,
        history_mode=fold.definition.history_mode,
        training_start=fold.definition.training_start,
        training_end=fold.definition.training_end,
        validation_start=fold.definition.validation_start,
        validation_end=fold.definition.validation_start + duration,
    )
    evaluations = evaluate_combination(
        dataset=cast(MLExperimentDataset, dataset),
        engine_name=ALL_ENGINES[0],
        duration=duration,
        configuration=DEFAULT_SCREENING_CONFIGURATION,
    )
    assert all(item.validation_row_count <= 1 for item in evaluations[:1])


def test_last_2023_fold_bounds_long_targets_before_final_test() -> None:
    start = datetime(2023, 12, 15, tzinfo=UTC)
    boundaries = np.asarray(
        [
            int((start + timedelta(minutes=30 * index)).timestamp() * 1e9)
            for index in range(34 * 48)
        ],
        dtype=np.int64,
    )
    validation_start = datetime(2023, 12, 20, tzinfo=UTC)
    validation_end = datetime(2024, 1, 1, tzinfo=UTC)
    definition = TemporalFoldDefinition(
        "last-2023",
        TrainingHistoryMode.ACCUMULATING,
        start,
        validation_start - timedelta(days=7),
        validation_start,
        validation_end,
    )
    validation = np.flatnonzero(
        (boundaries >= int(validation_start.timestamp() * 1e9))
        & (boundaries < int(validation_end.timestamp() * 1e9))
    )
    dataset = SimpleNamespace(boundary_time_ns=boundaries)
    bounded = _bounded_fold_indices(
        dataset=cast(MLExperimentDataset, dataset),
        definition=definition,
        training_indices=np.asarray([], dtype=np.int64),
        validation_indices=validation,
        duration=timedelta(days=7),
    )
    completion = boundaries[bounded.validation_indices] + int(
        timedelta(days=7).total_seconds() * 1e9
    )
    assert len(completion) > 0
    assert np.all(completion <= int(validation_end.timestamp() * 1e9))
    assert boundaries[validation[-1]] + int(
        timedelta(days=7).total_seconds() * 1e9
    ) > int(validation_end.timestamp() * 1e9)


def test_final_test_target_mutation_cannot_change_development_metrics() -> None:
    first = _SyntheticDataset()
    second = _SyntheticDataset()
    second.targets.values[second.final_test_indices] = np.asarray(
        [1_000_000, -1_000_000], dtype=np.float32
    )
    first_result = evaluate_combination(
        dataset=cast(MLExperimentDataset, first),
        engine_name=ScreeningEngine.RIDGE,
        duration=timedelta(minutes=30),
        configuration=DEFAULT_SCREENING_CONFIGURATION,
    )
    second_result = evaluate_combination(
        dataset=cast(MLExperimentDataset, second),
        engine_name=ScreeningEngine.RIDGE,
        duration=timedelta(minutes=30),
        configuration=DEFAULT_SCREENING_CONFIGURATION,
    )
    assert [item.metrics for item in first_result] == [
        item.metrics for item in second_result
    ]


def test_all_contract_durations_and_experiment_key_identity() -> None:
    assert len(FUTURE_DURATIONS) == 80
    instrument = MARKET_INSTRUMENTS[0].instrument
    durations = tuple(int(value.total_seconds()) for value in FUTURE_DURATIONS)
    versions = {"scikit-learn": "1", "lightgbm": "2", "catboost": "3"}
    first = experiment_key(
        dataset_fingerprint="a" * 64,
        instrument=instrument,
        engines=ALL_ENGINES,
        durations=durations,
        configuration=DEFAULT_SCREENING_CONFIGURATION,
        versions=versions,
    )
    assert first == experiment_key(
        dataset_fingerprint="a" * 64,
        instrument=instrument,
        engines=ALL_ENGINES,
        durations=durations,
        configuration=DEFAULT_SCREENING_CONFIGURATION,
        versions=versions,
    )
    changed = experiment_key(
        dataset_fingerprint="b" * 64,
        instrument=instrument,
        engines=ALL_ENGINES,
        durations=durations,
        configuration=DEFAULT_SCREENING_CONFIGURATION,
        versions=versions,
    )
    assert first != changed
