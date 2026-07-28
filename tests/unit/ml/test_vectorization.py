from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import app.ml.dataset as dataset_module
from app.ml.dataset import LazyTargetProvider
from app.ml.gaps import CANDLE_INTERVAL_NS, continuity_prefix, datetime_to_ns
from app.ml.registry import VARIABLE_REGISTRY, VariableDefinition


@pytest.mark.parametrize("missing_position", [1, 3, 5])
def test_vectorized_target_continuity_rejects_gap_anywhere(
    missing_position: int,
) -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    complete = np.arange(7, dtype=np.int64) * CANDLE_INTERVAL_NS + datetime_to_ns(start)
    opens = np.delete(complete, missing_position)
    provider = LazyTargetProvider(
        boundaries_ns=np.asarray([complete[0] + CANDLE_INTERVAL_NS]),
        reference_indices=np.asarray([0], dtype=np.int64),
        open_time_ns=opens,
        closes=np.arange(len(opens), dtype=np.float64) + 100,
        feature_eligible=np.asarray([True], dtype=np.bool_),
        continuity=continuity_prefix(opens),
    )
    target = provider.for_duration(timedelta(minutes=30))
    assert not target.eligible[0]
    assert target.exclusions.as_dict()["missing_future_interval_candle"] == 1


def test_feature_builder_invokes_calculation_once_per_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = dataset_module._feature_series

    def counted(
        definition: VariableDefinition, cache: dataset_module._SeriesCache
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        nonlocal calls
        calls += 1
        return original(definition, cache)

    monkeypatch.setattr(dataset_module, "_feature_series", counted)
    from tests.unit.ml.test_dataset import INSTRUMENT, candles, columns

    dataset_module.build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(candles())
    )
    assert calls == len(VARIABLE_REGISTRY)
