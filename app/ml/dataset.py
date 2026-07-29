
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from app.market.catalog import Instrument

from .contract import (
    DEVELOPMENT_START,
    FINAL_TEST_END,
    FINAL_TEST_START,
    FUTURE_DURATIONS,
    MISSING_DATA_POLICY_VERSION,
    ML_EXPERIMENT_CONTRACT_VERSION,
    VARIABLE_REGISTRY_VERSION,
    TemporalFoldDefinition,
    temporal_fold_definitions,
)
from .errors import MLDatasetError
from .gaps import (
    CANDLE_INTERVAL_NS,
    GapSummary,
    continuity_prefix,
    datetime_to_ns,
    ns_to_datetime,
    summarize_gaps,
)
from .reader import MLCandleColumns
from .registry import MAXIMUM_HISTORICAL_LOOKBACK, VARIABLE_REGISTRY, VariableDefinition


@dataclass(frozen=True, slots=True)
class ExclusionCounts:
    counts: tuple[tuple[str, int], ...]

    @classmethod
    def from_counter(cls, counter: Counter[str]) -> "ExclusionCounts":
        return cls(tuple(sorted(counter.items())))

    def as_dict(self) -> dict[str, int]:
        return dict(self.counts)

    @property
    def total(self) -> int:
        return sum(value for _, value in self.counts)


@dataclass(frozen=True, slots=True)
class TemporalFoldManifest:
    definition: TemporalFoldDefinition
    training_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EligibleFoldIndices:
    training_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TargetVector:
    duration: timedelta
    values: NDArray[np.float32]
    eligible: NDArray[np.bool_]
    exclusions: ExclusionCounts


@dataclass(frozen=True, slots=True)
class MLDatasetMetadata:
    instrument: Instrument
    requested_start: datetime
    requested_end: datetime
    actual_start: datetime | None
    actual_end: datetime | None
    candle_count: int
    gaps: GapSummary
    feature_count: int
    maximum_historical_lookback: timedelta
    fingerprint: str


class LazyTargetProvider:
    """Vectorize one target duration on demand without retaining other durations."""

    def __init__(
        self,
        *,
        boundaries_ns: NDArray[np.int64],
        reference_indices: NDArray[np.int64],
        open_time_ns: NDArray[np.int64],
        closes: NDArray[np.float64],
        feature_eligible: NDArray[np.bool_],
        continuity: NDArray[np.int64],
    ) -> None:
        self._boundaries_ns = boundaries_ns
        self._reference_indices = reference_indices
        self._open_time_ns = open_time_ns
        self._closes = closes
        self._feature_eligible = feature_eligible
        self._continuity = continuity

    def for_duration(
        self,
        duration: timedelta,
        *,
        candidate_indices: NDArray[np.int64] | None = None,
    ) -> TargetVector:
        if duration not in FUTURE_DURATIONS:
            raise MLDatasetError("duration is not part of contract version 1")
        row_count = len(self._boundaries_ns)
        values = np.full(row_count, np.nan, dtype=np.float32)
        eligible = np.zeros(row_count, dtype=np.bool_)
        reasons = np.full(row_count, "", dtype="U32")
        reasons[~self._feature_eligible] = "feature_ineligible"
        candidates = self._feature_eligible & (self._reference_indices >= 0)
        if candidate_indices is not None:
            requested = np.zeros(row_count, dtype=np.bool_)
            requested[candidate_indices] = True
            candidates &= requested
        duration_ns = int(duration.total_seconds() * 1_000_000_000)
        target_ns = self._boundaries_ns + duration_ns - CANDLE_INTERVAL_NS
        positions = np.searchsorted(self._open_time_ns, target_ns)
        bounded = positions < len(self._open_time_ns)
        exact = np.zeros(row_count, dtype=np.bool_)
        exact[bounded] = self._open_time_ns[positions[bounded]] == target_ns[bounded]
        missing_target = candidates & ~exact
        reasons[missing_target] = "missing_future_target_candle"
        exact_candidates = candidates & exact
        expected_steps = duration_ns // CANDLE_INTERVAL_NS
        expected_position = self._reference_indices + expected_steps
        position_matches = positions == expected_position
        continuity_matches = np.zeros(row_count, dtype=np.bool_)
        valid_positions = exact_candidates & position_matches
        rows = np.flatnonzero(valid_positions)
        continuity_matches[rows] = (
            self._continuity[positions[rows]]
            == self._continuity[self._reference_indices[rows]]
        )
        interval_complete = valid_positions & continuity_matches
        reasons[exact_candidates & ~interval_complete] = (
            "missing_future_interval_candle"
        )
        rows = np.flatnonzero(interval_complete)
        calculated = (
            self._closes[positions[rows]] / self._closes[self._reference_indices[rows]]
            - 1.0
        )
        finite = np.isfinite(calculated)
        values[rows[finite]] = calculated[finite].astype(np.float32)
        eligible[rows[finite]] = True
        reasons[rows[~finite]] = "non_finite_target"
        counter = Counter(str(reason) for reason in reasons if reason)
        return TargetVector(
            duration, values, eligible, ExclusionCounts.from_counter(counter)
        )


@dataclass(frozen=True, slots=True)
class MLExperimentDataset:
    metadata: MLDatasetMetadata
    boundary_time_ns: NDArray[np.int64]
    features: NDArray[np.float32]
    feature_eligible: NDArray[np.bool_]
    feature_exclusions: ExclusionCounts
    folds: tuple[TemporalFoldManifest, ...]
    final_test_indices: NDArray[np.int64]
    targets: LazyTargetProvider

    @property
    def boundaries(self) -> tuple[datetime, ...]:
        return tuple(ns_to_datetime(int(value)) for value in self.boundary_time_ns)

    def eligible_indices(
        self, *, fold_name: str, duration: timedelta
    ) -> EligibleFoldIndices:
        fold = next(
            (item for item in self.folds if item.definition.name == fold_name), None
        )
        if fold is None:
            raise MLDatasetError(f"unknown temporal fold: {fold_name}")
        target = self.targets.for_duration(duration)
        eligible = self.feature_eligible & target.eligible
        return EligibleFoldIndices(
            fold.training_indices[eligible[fold.training_indices]],
            fold.validation_indices[eligible[fold.validation_indices]],
        )


def _rolling_sum(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    result = np.full(len(values), np.nan)
    if window <= len(values):
        cumulative = np.concatenate((np.zeros(1), np.cumsum(values)))
        result[window - 1 :] = cumulative[window:] - cumulative[:-window]
    return result


def _safe_ratio(numerator: ArrayLike, denominator: ArrayLike) -> NDArray[np.float64]:
    """Divide finite values without warnings and preserve undefined ratios as NaN.

    A finite 0/0 pair represents neutral relative activity and returns 0. A
    finite non-zero numerator over zero, or any non-finite input, remains NaN.
    """

    numerator_array, denominator_array = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64),
        np.asarray(denominator, dtype=np.float64),
    )
    result = np.full(numerator_array.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(numerator_array) & np.isfinite(denominator_array)
    divisible = finite & (denominator_array != 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(
            numerator_array,
            denominator_array,
            out=result,
            where=divisible,
        )
    result[~np.isfinite(result)] = np.nan
    result[finite & (numerator_array == 0) & (denominator_array == 0)] = 0.0
    return result


def _safe_percentage_change(
    recent: ArrayLike, previous: ArrayLike
) -> NDArray[np.float64]:
    """Return a safe relative change while keeping a finite 0/0 pair neutral."""

    recent_array, previous_array = np.broadcast_arrays(
        np.asarray(recent, dtype=np.float64),
        np.asarray(previous, dtype=np.float64),
    )
    ratio = _safe_ratio(recent_array, previous_array)
    both_zero = (
        np.isfinite(recent_array)
        & np.isfinite(previous_array)
        & (recent_array == 0)
        & (previous_array == 0)
    )
    result = ratio - 1.0
    result[both_zero] = 0.0
    return result


def _rolling_mean(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    return _rolling_sum(values, window) / window


def _rolling_std(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    mean = _rolling_mean(values, window)
    square_mean = _rolling_mean(values * values, window)
    return np.sqrt(np.maximum(square_mean - mean * mean, 0.0))


def _rolling_extreme(
    values: NDArray[np.float64], window: int, *, maximum: bool
) -> NDArray[np.float64]:
    result = np.full(len(values), np.nan)
    if window <= len(values):
        views = sliding_window_view(values, window)
        reduced = np.max(views, axis=1) if maximum else np.min(views, axis=1)
        result[window - 1 :] = reduced
    return result


def _ema_series(
    values: NDArray[np.float64],
    period: int,
    continuity: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return a bounded finite-window EMA reset at every candle discontinuity.

    The first complete window uses the explicit finite-window weights. Later
    consecutive windows use the algebraically equivalent O(1) recurrence, so
    no row rebuilds its weight array and no pre-gap value can enter a new segment.
    """

    result = np.full(len(values), np.nan)
    if period < 1 or period > len(values) or len(continuity) != len(values):
        return result
    alpha = 2.0 / (period + 1.0)
    q = 1.0 - alpha
    q_to_period = q**period
    weights = alpha * np.power(q, np.arange(period - 1, -1, -1))
    weights[0] += q_to_period
    segment_start = 0
    for index in range(len(values)):
        if index and continuity[index] != continuity[index - 1]:
            segment_start = index
        segment_length = index - segment_start + 1
        if segment_length < period:
            continue
        if segment_length == period:
            result[index] = float(
                np.dot(values[index - period + 1 : index + 1], weights)
            )
            continue
        result[index] = (
            q * result[index - 1]
            + alpha * values[index]
            + q_to_period * (values[index - period + 1] - values[index - period])
        )
    return result


class _SeriesCache:
    """Reuse causal full-series rolling calculations across registry variables."""

    def __init__(self, columns: MLCandleColumns, continuity: NDArray[np.int64]) -> None:
        self.columns = columns
        self.continuity = continuity
        self.returns = np.full(len(columns), np.nan)
        self.returns[1:] = columns.close_price[1:] / columns.close_price[:-1] - 1.0
        self._cache: dict[tuple[str, int], NDArray[np.float64]] = {}

    def cached(self, operation: str, period: int) -> NDArray[np.float64]:
        key = (operation, period)
        if key in self._cache:
            return self._cache[key]
        source = self.columns.close_price
        if operation == "ema":
            value = _ema_series(source, period, self.continuity)
        elif operation == "return_std":
            value = _rolling_std(np.nan_to_num(self.returns), period)
        elif operation == "high":
            value = _rolling_extreme(self.columns.high_price, period, maximum=True)
        elif operation == "low":
            value = _rolling_extreme(self.columns.low_price, period, maximum=False)
        elif operation == "volume_sum":
            value = _rolling_sum(self.columns.base_asset_volume, period)
        elif operation == "volume_mean":
            value = _rolling_mean(self.columns.base_asset_volume, period)
        else:
            raise MLDatasetError(f"unknown series operation: {operation}")
        self._cache[key] = value
        return value


def _invalidate_window(
    values: NDArray[np.float64], continuity: NDArray[np.int64], window: int
) -> NDArray[np.float64]:
    result = values.copy()
    indices = np.arange(len(values))
    starts = indices - window + 1
    valid = starts >= 0
    rows = np.flatnonzero(valid)
    valid[rows] &= continuity[rows] == continuity[starts[rows]]
    result[~valid] = np.nan
    return result


def _feature_series(
    definition: VariableDefinition, cache: _SeriesCache
) -> NDArray[np.float64]:
    columns = cache.columns
    close = columns.close_price
    calculation, _, argument = definition.calculation.partition(":")
    count = max(1, int(definition.lookback.total_seconds() // 300))
    period = count - 1 if calculation in {"return", "ema_slope"} else count
    if calculation == "return":
        result = close / np.roll(close, period) - 1.0
    elif calculation.startswith("ema_") or calculation == "trend_normalized":
        ema = cache.cached("ema", period)
        result = close / ema - 1.0
        if calculation == "ema_slope":
            result = ema / np.roll(ema, 1) - 1.0
        elif calculation == "ema_spread":
            result = cache.cached("ema", int(argument) // 5) / ema - 1.0
        elif calculation == "trend_normalized":
            denominator = cache.cached("return_std", period)
            result = np.divide(
                result, denominator, out=np.zeros(len(close)), where=denominator != 0
            )
    elif calculation in {
        "previous_high",
        "previous_low",
        "range_position",
        "distance_from_high",
        "distance_from_low",
        "breakout_high",
        "breakout_low",
    }:
        previous_high = np.roll(cache.cached("high", period - 1), 1)
        previous_low = np.roll(cache.cached("low", period - 1), 1)
        span = previous_high - previous_low
        if calculation == "previous_high":
            result = previous_high
        elif calculation == "previous_low":
            result = previous_low
        elif calculation == "range_position":
            result = np.divide(
                close - previous_low,
                span,
                out=np.full(len(close), 0.5),
                where=span != 0,
            )
        elif calculation == "distance_from_high":
            result = close / previous_high - 1.0
        elif calculation == "distance_from_low":
            result = close / previous_low - 1.0
        elif calculation == "breakout_high":
            result = (close > previous_high).astype(float)
        else:
            result = (close < previous_low).astype(float)
    elif calculation == "return_volatility":
        result = cache.cached("return_std", period)
    elif calculation == "normalized_range":
        result = _rolling_mean((columns.high_price - columns.low_price) / close, period)
    elif calculation == "volatility_change":
        half = period // 2
        result = cache.cached("return_std", half) - np.roll(
            cache.cached("return_std", half), half
        )
    elif calculation == "volatility_ratio":
        denominator = cache.cached("return_std", period)
        result = np.divide(
            cache.cached("return_std", int(argument) // 5),
            denominator,
            out=np.zeros(len(close)),
            where=denominator != 0,
        )
    elif calculation == "relative_volume":
        baseline = np.roll(cache.cached("volume_mean", period - 1), 1)
        result = _safe_ratio(columns.base_asset_volume, baseline)
    elif calculation in {"volume_change", "normalized_volume"}:
        half = period // 2
        recent = cache.cached("volume_sum", half)
        old = np.roll(recent, half)
        result = _safe_percentage_change(recent, old)
    elif calculation == "signed_price_volume":
        result = _rolling_mean(
            np.nan_to_num(cache.returns) * columns.base_asset_volume, period
        )
    elif calculation == "candle_body":
        result = np.abs(close - columns.open_price) / columns.open_price
    elif calculation == "candle_range":
        result = (columns.high_price - columns.low_price) / columns.open_price
    elif calculation == "upper_wick":
        result = (
            columns.high_price - np.maximum(columns.open_price, close)
        ) / columns.open_price
    elif calculation == "lower_wick":
        result = (
            np.minimum(columns.open_price, close) - columns.low_price
        ) / columns.open_price
    elif calculation == "candle_close_position":
        span = columns.high_price - columns.low_price
        result = np.divide(
            close - columns.low_price,
            span,
            out=np.full(len(close), 0.5),
            where=span != 0,
        )
    elif calculation == "positive_fraction":
        result = _rolling_mean((close > columns.open_price).astype(float), period)
    else:
        timestamps = columns.open_time_ns.astype("datetime64[ns]")
        days = timestamps.astype("datetime64[D]")
        seconds = (timestamps - days).astype("timedelta64[s]").astype(float)
        weekdays = (days.astype(np.int64) + 3) % 7
        if calculation == "utc_time_sin":
            result = np.sin(2 * math.pi * seconds / 86400)
        elif calculation == "utc_time_cos":
            result = np.cos(2 * math.pi * seconds / 86400)
        elif calculation == "utc_weekday_sin":
            result = np.sin(2 * math.pi * weekdays / 7)
        elif calculation == "utc_weekday_cos":
            result = np.cos(2 * math.pi * weekdays / 7)
        else:
            raise MLDatasetError(f"unsupported variable calculation: {calculation}")
    return _invalidate_window(
        np.asarray(result, dtype=np.float64), cache.continuity, count
    )


def _fingerprint(
    instrument: Instrument, columns: MLCandleColumns, requested_end: datetime
) -> str:
    digest = hashlib.sha256()
    contract = {
        "contract": ML_EXPERIMENT_CONTRACT_VERSION,
        "registry": VARIABLE_REGISTRY_VERSION,
        "missing_policy": MISSING_DATA_POLICY_VERSION,
        "instrument": [
            instrument.asset.value,
            instrument.provider.value,
            instrument.market_type.value,
            instrument.interval.value,
        ],
        "range": [DEVELOPMENT_START.isoformat(), requested_end.isoformat()],
        "variables": [
            (item.name, item.calculation, item.lookback.total_seconds())
            for item in VARIABLE_REGISTRY
        ],
        "durations": [item.total_seconds() for item in FUTURE_DURATIONS],
        "folds": [
            (
                item.name,
                item.training_start.isoformat(),
                item.training_end.isoformat(),
                item.validation_start.isoformat(),
                item.validation_end.isoformat(),
            )
            for item in temporal_fold_definitions()
        ],
    }
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    for array in (
        columns.open_time_ns,
        columns.close_time_ns,
        columns.open_price,
        columns.high_price,
        columns.low_price,
        columns.close_price,
        columns.base_asset_volume,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_ml_experiment_dataset(
    *,
    instrument: Instrument,
    columns: MLCandleColumns,
    requested_end: datetime = FINAL_TEST_END,
) -> MLExperimentDataset:
    if requested_end not in {FINAL_TEST_START, FINAL_TEST_END}:
        raise MLDatasetError(
            "ML dataset end must be the development or protected final-test end"
        )
    if len(columns) and (
        np.any(np.diff(columns.open_time_ns) <= 0)
        or np.any(columns.close_time_ns >= columns.open_time_ns + CANDLE_INTERVAL_NS)
    ):
        raise MLDatasetError("candle timestamps violate ordering or closure semantics")
    start_ns = datetime_to_ns(DEVELOPMENT_START)
    end_ns = datetime_to_ns(requested_end)
    selected_mask = (columns.open_time_ns >= start_ns) & (columns.open_time_ns < end_ns)
    selected = columns
    if not np.all(selected_mask):
        selected = MLCandleColumns(
            columns.open_time_ns[selected_mask],
            columns.close_time_ns[selected_mask],
            columns.open_price[selected_mask],
            columns.high_price[selected_mask],
            columns.low_price[selected_mask],
            columns.close_price[selected_mask],
            columns.base_asset_volume[selected_mask],
        )
    continuity = continuity_prefix(selected.open_time_ns)
    gaps = summarize_gaps(selected.open_time_ns)
    boundary_ns = np.arange(start_ns, end_ns, 30 * 60 * 1_000_000_000, dtype=np.int64)
    reference_ns = boundary_ns - CANDLE_INTERVAL_NS
    reference_indices = np.searchsorted(selected.open_time_ns, reference_ns).astype(
        np.int64
    )
    bounded = reference_indices < len(selected)
    exact = np.zeros(len(boundary_ns), dtype=np.bool_)
    exact[bounded] = (
        selected.open_time_ns[reference_indices[bounded]] == reference_ns[bounded]
    )
    closed = np.zeros(len(boundary_ns), dtype=np.bool_)
    rows = np.flatnonzero(exact)
    closed[rows] = selected.close_time_ns[reference_indices[rows]] < boundary_ns[rows]
    features = np.full(
        (len(boundary_ns), len(VARIABLE_REGISTRY)), np.nan, dtype=np.float32
    )
    rows = np.flatnonzero(exact & closed)
    cache = _SeriesCache(selected, continuity)
    for column, definition in enumerate(VARIABLE_REGISTRY):
        series = _feature_series(definition, cache)
        features[rows, column] = series[reference_indices[rows]].astype(np.float32)
        del series
    feature_eligible = np.isfinite(features).all(axis=1)
    reason_by_row = np.full(len(boundary_ns), "", dtype="U32")
    reason_by_row[~exact] = "missing_reference_candle"
    reason_by_row[exact & ~closed] = "reference_candle_not_closed"
    candidates = exact & closed
    maximum_steps = int(MAXIMUM_HISTORICAL_LOOKBACK.total_seconds() // 300)
    enough = reference_indices >= maximum_steps - 1
    reason_by_row[candidates & ~enough] = "insufficient_history"
    complete = np.zeros(len(boundary_ns), dtype=np.bool_)
    rows = np.flatnonzero(candidates & enough)
    starts = reference_indices[rows] - maximum_steps + 1
    complete[rows] = continuity[reference_indices[rows]] == continuity[starts]
    reason_by_row[candidates & enough & ~complete] = "missing_historical_candle"
    reason_by_row[candidates & enough & complete & ~feature_eligible] = (
        "non_finite_feature"
    )
    reasons = Counter(str(reason) for reason in reason_by_row if reason)
    folds = tuple(
        TemporalFoldManifest(
            definition,
            np.flatnonzero(
                (boundary_ns >= datetime_to_ns(definition.training_start))
                & (boundary_ns < datetime_to_ns(definition.training_end))
            ).astype(np.int64),
            np.flatnonzero(
                (boundary_ns >= datetime_to_ns(definition.validation_start))
                & (boundary_ns < datetime_to_ns(definition.validation_end))
            ).astype(np.int64),
        )
        for definition in temporal_fold_definitions()
    )
    final_indices = np.flatnonzero(
        (boundary_ns >= datetime_to_ns(FINAL_TEST_START)) & (boundary_ns < end_ns)
    ).astype(np.int64)
    metadata = MLDatasetMetadata(
        instrument,
        DEVELOPMENT_START,
        requested_end,
        ns_to_datetime(int(selected.open_time_ns[0])) if len(selected) else None,
        ns_to_datetime(int(selected.open_time_ns[-1])) if len(selected) else None,
        len(selected),
        gaps,
        len(VARIABLE_REGISTRY),
        MAXIMUM_HISTORICAL_LOOKBACK,
        _fingerprint(instrument, selected, requested_end),
    )
    targets = LazyTargetProvider(
        boundaries_ns=boundary_ns,
        reference_indices=reference_indices,
        open_time_ns=selected.open_time_ns,
        closes=selected.close_price,
        feature_eligible=feature_eligible,
        continuity=continuity,
    )
    return MLExperimentDataset(
        metadata,
        boundary_ns,
        features,
        feature_eligible,
        ExclusionCounts.from_counter(reasons),
        folds,
        final_indices,
        targets,
    )
