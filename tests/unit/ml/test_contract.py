from datetime import UTC, datetime, timedelta

from app.ml.contract import (
    DEVELOPMENT_END,
    FINAL_TEST_START,
    FUTURE_DURATIONS,
    SPLIT_GAP,
    TrainingHistoryMode,
    temporal_fold_definitions,
)
from app.ml.registry import VARIABLE_REGISTRY, build_variable_registry


def test_future_duration_contract_is_exact_and_algorithmic() -> None:
    assert len(FUTURE_DURATIONS) == 80
    assert len(set(FUTURE_DURATIONS)) == 80
    assert tuple(sorted(FUTURE_DURATIONS)) == FUTURE_DURATIONS
    assert FUTURE_DURATIONS[0] == timedelta(minutes=30)
    assert FUTURE_DURATIONS[-1] == timedelta(days=7)


def test_variable_registry_is_stable_and_float32() -> None:
    rebuilt = build_variable_registry()
    assert rebuilt == VARIABLE_REGISTRY
    assert tuple(item.name for item in rebuilt) == tuple(
        sorted(item.name for item in rebuilt)
    )
    assert len({item.name for item in rebuilt}) == len(rebuilt)
    assert all(str(item.output_type) == "float32" for item in rebuilt)


def test_temporal_folds_are_chronological_gapped_and_never_use_final_test() -> None:
    folds = temporal_fold_definitions()
    assert folds
    for fold in folds:
        assert fold.training_start < fold.training_end
        assert fold.training_end + SPLIT_GAP == fold.validation_start
        assert fold.validation_start < fold.validation_end <= DEVELOPMENT_END
        assert fold.validation_end <= FINAL_TEST_START
    rolling_two = [
        fold
        for fold in folds
        if fold.history_mode is TrainingHistoryMode.ROLLING_TWO_YEARS
    ]
    assert rolling_two
    assert all(
        fold.training_start.replace(year=fold.training_start.year + 2)
        == fold.training_end
        for fold in rolling_two
    )
    assert not any(
        fold.history_mode is TrainingHistoryMode.ROLLING_THREE_YEARS
        and fold.validation_start == datetime(2022, 1, 1, tzinfo=UTC)
        for fold in folds
    )
