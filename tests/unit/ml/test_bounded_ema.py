from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import app.ml.dataset as dataset_module
from app.ml.dataset import _ema_series, _feature_series, _SeriesCache
from app.ml.gaps import continuity_prefix, datetime_to_ns
from app.ml.reader import MLCandleColumns
from app.ml.registry import VARIABLE_REGISTRY


def explicit_finite_ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(values), np.nan)
    alpha = 2.0 / (period + 1.0)
    q = 1.0 - alpha
    weights = alpha * np.power(q, np.arange(period - 1, -1, -1))
    weights[0] += q**period
    for index in range(period - 1, len(values)):
        result[index] = np.dot(values[index - period + 1 : index + 1], weights)
    return result


@pytest.mark.parametrize("period", [2, 3, 5, 8])
def test_optimized_bounded_ema_matches_explicit_reference(period: int) -> None:
    values = np.asarray(
        [100, 102, 101, 106, 104, 108, 109, 105, 111, 114, 113],
        dtype=np.float64,
    )
    continuity = np.zeros(len(values), dtype=np.int64)
    np.testing.assert_allclose(
        _ema_series(values, period, continuity),
        explicit_finite_ema(values, period),
        rtol=1e-13,
        atol=1e-13,
        equal_nan=True,
    )


def make_columns(open_indices: list[int], closes: np.ndarray) -> MLCandleColumns:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    opens = np.asarray(
        [datetime_to_ns(start + timedelta(minutes=5 * item)) for item in open_indices],
        dtype=np.int64,
    )
    interval_ns = int(timedelta(minutes=5).total_seconds() * 1_000_000_000)
    return MLCandleColumns(
        opens,
        opens + interval_ns - 1_000_000,
        closes - 0.25,
        closes + 1,
        closes - 1,
        closes,
        np.full(len(closes), 10.0),
    )


def test_pre_gap_prices_cannot_influence_post_gap_ema_features() -> None:
    indices = [*range(15), *range(16, 32)]
    post_gap = np.linspace(200, 215, 16)
    first = make_columns(indices, np.concatenate((np.linspace(100, 114, 15), post_gap)))
    second = make_columns(
        indices, np.concatenate((np.linspace(900, 914, 15), post_gap))
    )
    definition = next(
        item for item in VARIABLE_REGISTRY if item.name == "ema_distance_1h"
    )
    first_values = _feature_series(
        definition, _SeriesCache(first, continuity_prefix(first.open_time_ns))
    )
    second_values = _feature_series(
        definition, _SeriesCache(second, continuity_prefix(second.open_time_ns))
    )
    post_gap_start = 15
    assert np.isnan(first_values[post_gap_start : post_gap_start + 11]).all()
    assert np.isnan(second_values[post_gap_start : post_gap_start + 11]).all()
    np.testing.assert_allclose(
        first_values[post_gap_start + 11 :],
        second_values[post_gap_start + 11 :],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("current_position", "expected_finite"),
    [(9, True), (7, False), (5, False)],
    ids=["gap-before-window", "gap-inside-window", "gap-before-current"],
)
def test_gap_position_controls_bounded_ema_eligibility(
    current_position: int, expected_finite: bool
) -> None:
    indices = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    columns = make_columns(indices, np.arange(len(indices), dtype=np.float64) + 100)
    result = _ema_series(
        columns.close_price, 4, continuity_prefix(columns.open_time_ns)
    )
    assert bool(np.isfinite(result[current_position])) is expected_finite


def test_ema_is_calculated_once_per_cached_series_and_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = make_columns(list(range(20)), np.arange(20, dtype=np.float64) + 100)
    continuity = continuity_prefix(columns.open_time_ns)
    calls = 0
    original = dataset_module._ema_series

    def counted(values: np.ndarray, period: int, prefix: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(values, period, prefix)

    monkeypatch.setattr(dataset_module, "_ema_series", counted)
    cache = _SeriesCache(columns, continuity)
    assert cache.cached("ema", 4) is cache.cached("ema", 4)
    cache.cached("ema", 8)
    assert calls == 2
