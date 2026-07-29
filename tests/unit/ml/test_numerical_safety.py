import warnings
from datetime import timedelta

import numpy as np
import pytest

from app.ml.contract import ML_EXPERIMENT_CONTRACT_VERSION
from app.ml.dataset import (
    _feature_series,
    _safe_percentage_change,
    _safe_ratio,
    _SeriesCache,
    build_ml_experiment_dataset,
)
from app.ml.gaps import continuity_prefix
from app.ml.registry import VARIABLE_REGISTRY
from tests.unit.ml.test_dataset import INSTRUMENT, candles, columns


def test_safe_ratio_has_explicit_finite_zero_and_invalid_semantics() -> None:
    numerator = np.asarray([6, 0, 2, -2, np.nan, 2, np.inf, 2], dtype=np.float64)
    denominator = np.asarray([3, 0, 0, 0, 2, np.nan, 2, np.inf], dtype=np.float64)
    result = _safe_ratio(numerator, denominator)
    assert result[0] == 2
    assert result[1] == 0
    assert np.isnan(result[2:]).all()
    assert not np.isinf(result).any()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_safe_ratio_converts_finite_overflow_to_nan_without_warning(
    sign: float,
) -> None:
    numerator = np.asarray([sign * np.finfo(np.float64).max])
    denominator = np.asarray([np.finfo(np.float64).tiny])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _safe_ratio(numerator, denominator)
    assert np.isnan(result[0])
    assert not np.isinf(result).any()


@pytest.mark.parametrize(
    ("calculation", "current_volume", "expected"),
    [
        ("relative_volume", 0.0, 0.0),
        ("relative_volume", 5.0, np.nan),
        ("volume_change", 0.0, 0.0),
        ("volume_change", 5.0, np.nan),
        ("normalized_volume", 0.0, 0.0),
        ("normalized_volume", 5.0, np.nan),
    ],
)
def test_volume_ratios_are_safe_and_warning_free(
    calculation: str, current_volume: float, expected: float
) -> None:
    source = columns(candles())
    source.base_asset_volume[:] = 0
    definition = next(
        item
        for item in VARIABLE_REGISTRY
        if item.calculation == calculation and item.name.endswith("_1h")
    )
    if calculation == "relative_volume":
        source.base_asset_volume[-1] = current_volume
    else:
        source.base_asset_volume[-12:] = current_volume
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        values = _feature_series(
            definition,
            _SeriesCache(source, continuity_prefix(source.open_time_ns)),
        )
    if np.isnan(expected):
        assert np.isnan(values[-1])
    else:
        assert values[-1] == expected
    assert not np.isinf(values).any()


def test_safe_percentage_change_keeps_zero_over_zero_neutral() -> None:
    result = _safe_percentage_change(
        np.asarray([0, 5, 10], dtype=np.float64),
        np.asarray([0, 0, 5], dtype=np.float64),
    )
    assert result[0] == 0
    assert np.isnan(result[1])
    assert result[2] == 1


def test_dataset_excludes_a_genuinely_undefined_volume_ratio() -> None:
    source = columns(candles())
    source.base_asset_volume[:] = 0
    neutral = build_ml_experiment_dataset(instrument=INSTRUMENT, columns=source)
    row = int(np.flatnonzero(neutral.feature_eligible)[0])
    reference_open_ns = neutral.boundary_time_ns[row] - int(
        timedelta(minutes=5).total_seconds() * 1_000_000_000
    )
    reference_index = int(np.searchsorted(source.open_time_ns, reference_open_ns))
    source.base_asset_volume[reference_index] = 5
    undefined = build_ml_experiment_dataset(instrument=INSTRUMENT, columns=source)
    assert not undefined.feature_eligible[row]
    assert undefined.feature_exclusions.as_dict()["non_finite_feature"] >= 1


def test_contract_and_registry_identity_are_unchanged_except_version() -> None:
    assert ML_EXPERIMENT_CONTRACT_VERSION == 2
    assert len(VARIABLE_REGISTRY) == 126
