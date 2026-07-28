import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Final

import numpy as np
from numpy.typing import NDArray

from app.market.catalog import Instrument
from app.optimization.historical_data import HistoricalCandle

from .contract import (
    CANDLE_INTERVAL,
    DECISION_INTERVAL,
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
from .registry import MAXIMUM_HISTORICAL_LOOKBACK, VARIABLE_REGISTRY, VariableDefinition

NANOSECONDS_PER_SECOND: Final = 1_000_000_000


class MLDatasetError(ValueError):
    """Raised when input cannot satisfy the frozen ML dataset contract."""


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
    detected_gap_count: int
    feature_count: int
    maximum_historical_lookback: timedelta
    fingerprint: str


class LazyTargetProvider:
    """Calculate one target vector on demand, never an 80-column target matrix."""

    def __init__(
        self,
        *,
        boundaries: tuple[datetime, ...],
        reference_indices: NDArray[np.int64],
        opens: tuple[datetime, ...],
        closes: NDArray[np.float64],
        feature_eligible: NDArray[np.bool_],
    ) -> None:
        self._boundaries = boundaries
        self._reference_indices = reference_indices
        self._opens = opens
        self._closes = closes
        self._feature_eligible = feature_eligible
        self._index_by_open = {value: index for index, value in enumerate(opens)}

    def for_duration(self, duration: timedelta) -> TargetVector:
        if duration not in FUTURE_DURATIONS:
            raise MLDatasetError("duration is not part of contract version 1")
        values = np.full(len(self._boundaries), np.nan, dtype=np.float32)
        eligible = np.zeros(len(self._boundaries), dtype=np.bool_)
        exclusions: Counter[str] = Counter()
        steps = int(duration / CANDLE_INTERVAL)
        for row, (boundary, reference_index) in enumerate(
            zip(self._boundaries, self._reference_indices, strict=True)
        ):
            if not self._feature_eligible[row]:
                exclusions["feature_ineligible"] += 1
                continue
            target_open = boundary + duration - CANDLE_INTERVAL
            target_index = self._index_by_open.get(target_open)
            if target_index is None:
                exclusions["missing_future_target_candle"] += 1
                continue
            if target_index - int(reference_index) != steps:
                exclusions["missing_future_interval_candle"] += 1
                continue
            expected = boundary
            complete = True
            for index in range(int(reference_index) + 1, target_index + 1):
                if self._opens[index] != expected:
                    complete = False
                    break
                expected += CANDLE_INTERVAL
            if not complete:
                exclusions["missing_future_interval_candle"] += 1
                continue
            reference_close = self._closes[reference_index]
            value = self._closes[target_index] / reference_close - 1.0
            if not math.isfinite(value):
                exclusions["non_finite_target"] += 1
                continue
            values[row] = np.float32(value)
            eligible[row] = True
        return TargetVector(
            duration, values, eligible, ExclusionCounts.from_counter(exclusions)
        )


@dataclass(frozen=True, slots=True)
class MLExperimentDataset:
    metadata: MLDatasetMetadata
    boundaries: tuple[datetime, ...]
    features: NDArray[np.float32]
    feature_eligible: NDArray[np.bool_]
    feature_exclusions: ExclusionCounts
    folds: tuple[TemporalFoldManifest, ...]
    final_test_indices: NDArray[np.int64]
    targets: LazyTargetProvider


def _validate_candles(candles: tuple[HistoricalCandle, ...]) -> None:
    previous: datetime | None = None
    for candle in candles:
        if candle.open_time.tzinfo is not UTC or candle.close_time.tzinfo is not UTC:
            raise MLDatasetError("all candle timestamps must be timezone-aware UTC")
        if previous is not None and candle.open_time <= previous:
            raise MLDatasetError("candles must be strictly ordered and unique")
        if candle.close_time >= candle.open_time + CANDLE_INTERVAL:
            raise MLDatasetError(
                "candle close time is inconsistent with the 5m contract"
            )
        previous = candle.open_time


def _ema(values: NDArray[np.float64], period: int) -> float:
    """Calculate a finite-window EMA without a Python loop over every candle."""

    alpha = 2.0 / (period + 1.0)
    weights = np.power(1.0 - alpha, np.arange(len(values) - 1, -1, -1))
    weights *= alpha
    weights[0] += (1.0 - alpha) ** len(values)
    return float(np.dot(values, weights))


def _returns(closes: NDArray[np.float64]) -> NDArray[np.float64]:
    return closes[1:] / closes[:-1] - 1.0


def _calculate(
    definition: VariableDefinition,
    *,
    boundary: datetime,
    opens: NDArray[np.float64],
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
    volumes: NDArray[np.float64],
) -> float:
    calculation, _, argument = definition.calculation.partition(":")
    current = float(closes[-1])
    candle_count = max(1, int(definition.lookback / CANDLE_INTERVAL))
    window = closes[-candle_count:]
    if calculation == "return":
        return current / float(closes[0]) - 1.0
    if calculation.startswith("ema_") or calculation == "trend_normalized":
        period = max(
            1,
            int(
                (
                    definition.lookback
                    - (CANDLE_INTERVAL if calculation == "ema_slope" else timedelta())
                )
                / CANDLE_INTERVAL
            ),
        )
        ema = _ema(closes[-period:], period)
        if calculation == "ema_distance":
            return current / ema - 1.0
        if calculation == "ema_slope":
            previous = _ema(closes[-period - 1 : -1], period)
            return ema / previous - 1.0
        if calculation == "ema_spread":
            fast = int(argument) // 5
            return _ema(closes[-fast:], fast) / ema - 1.0
        volatility = float(np.std(_returns(window), ddof=0))
        return (current / ema - 1.0) / volatility if volatility else 0.0
    if calculation in {
        "previous_high",
        "previous_low",
        "range_position",
        "distance_from_high",
        "distance_from_low",
        "breakout_high",
        "breakout_low",
    }:
        previous_high = float(np.max(highs[:-1]))
        previous_low = float(np.min(lows[:-1]))
        span = previous_high - previous_low
        if calculation == "previous_high":
            return previous_high
        if calculation == "previous_low":
            return previous_low
        if calculation == "range_position":
            return (current - previous_low) / span if span else 0.5
        if calculation == "distance_from_high":
            return current / previous_high - 1.0
        if calculation == "distance_from_low":
            return current / previous_low - 1.0
        if calculation == "breakout_high":
            return float(current > previous_high)
        return float(current < previous_low)
    if calculation in {
        "return_volatility",
        "normalized_range",
        "volatility_change",
        "volatility_ratio",
    }:
        if calculation == "normalized_range":
            return float(np.mean((highs - lows) / closes))
        if calculation == "volatility_change":
            half = (len(closes) - 1) // 2
            first = float(np.std(_returns(closes[: half + 1])))
            second = float(np.std(_returns(closes[half:])))
            return second - first
        long_vol = float(np.std(_returns(closes)))
        if calculation == "return_volatility":
            return long_vol
        short_count = int(argument) // 5 + 1
        short_vol = float(np.std(_returns(closes[-short_count:])))
        return short_vol / long_vol if long_vol else 0.0
    if calculation in {
        "relative_volume",
        "volume_change",
        "normalized_volume",
        "signed_price_volume",
    }:
        if calculation == "relative_volume":
            baseline = float(np.mean(volumes[:-1]))
            return float(volumes[-1]) / baseline if baseline else 0.0
        if calculation in {"volume_change", "normalized_volume"}:
            half = len(volumes) // 2
            old = float(np.sum(volumes[:half]))
            new = float(np.sum(volumes[half:]))
            return new / old - 1.0 if old else 0.0
        return float(np.mean(_returns(closes) * volumes[1:]))
    open_price = float(opens[-1])
    high = float(highs[-1])
    low = float(lows[-1])
    close = float(closes[-1])
    if calculation == "candle_body":
        return abs(close - open_price) / open_price
    if calculation == "candle_range":
        return (high - low) / open_price
    if calculation == "upper_wick":
        return (high - max(open_price, close)) / open_price
    if calculation == "lower_wick":
        return (min(open_price, close) - low) / open_price
    if calculation == "candle_close_position":
        return (close - low) / (high - low) if high != low else 0.5
    if calculation == "positive_fraction":
        return float(np.mean(closes > opens))
    seconds = boundary.hour * 3600 + boundary.minute * 60
    if calculation == "utc_time_sin":
        return math.sin(2 * math.pi * seconds / 86400)
    if calculation == "utc_time_cos":
        return math.cos(2 * math.pi * seconds / 86400)
    if calculation == "utc_weekday_sin":
        return math.sin(2 * math.pi * boundary.weekday() / 7)
    if calculation == "utc_weekday_cos":
        return math.cos(2 * math.pi * boundary.weekday() / 7)
    raise MLDatasetError(f"unsupported variable calculation: {definition.calculation}")


def _fingerprint(instrument: Instrument, candles: tuple[HistoricalCandle, ...]) -> str:
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
        "range": [DEVELOPMENT_START.isoformat(), FINAL_TEST_END.isoformat()],
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
    for candle in candles:
        digest.update(
            "|".join(
                (
                    candle.open_time.isoformat(),
                    candle.close_time.isoformat(),
                    str(candle.open_price),
                    str(candle.high_price),
                    str(candle.low_price),
                    str(candle.close_price),
                    str(candle.base_asset_volume),
                )
            ).encode()
        )
    return digest.hexdigest()


def build_ml_experiment_dataset(
    *, instrument: Instrument, candles: tuple[HistoricalCandle, ...]
) -> MLExperimentDataset:
    _validate_candles(candles)
    selected = tuple(
        candle
        for candle in candles
        if DEVELOPMENT_START <= candle.open_time < FINAL_TEST_END
    )
    index_by_open = {candle.open_time: index for index, candle in enumerate(selected)}
    boundaries: list[datetime] = []
    boundary = DEVELOPMENT_START
    while boundary < FINAL_TEST_END:
        boundaries.append(boundary)
        boundary += DECISION_INTERVAL
    feature_matrix = np.full(
        (len(boundaries), len(VARIABLE_REGISTRY)), np.nan, dtype=np.float32
    )
    feature_eligible = np.zeros(len(boundaries), dtype=np.bool_)
    reference_indices = np.full(len(boundaries), -1, dtype=np.int64)
    exclusions: Counter[str] = Counter()
    opens = np.asarray([float(item.open_price) for item in selected], dtype=np.float64)
    highs = np.asarray([float(item.high_price) for item in selected], dtype=np.float64)
    lows = np.asarray([float(item.low_price) for item in selected], dtype=np.float64)
    closes = np.asarray(
        [float(item.close_price) for item in selected], dtype=np.float64
    )
    volumes = np.asarray(
        [float(item.base_asset_volume) for item in selected], dtype=np.float64
    )
    maximum_steps = int(MAXIMUM_HISTORICAL_LOOKBACK / CANDLE_INTERVAL)
    for row, current_boundary in enumerate(boundaries):
        reference_open = current_boundary - CANDLE_INTERVAL
        reference_index = index_by_open.get(reference_open)
        if reference_index is None:
            exclusions["missing_reference_candle"] += 1
            continue
        reference_indices[row] = reference_index
        if selected[reference_index].close_time >= current_boundary:
            exclusions["reference_candle_not_closed"] += 1
            continue
        start_index = reference_index - maximum_steps + 1
        if start_index < 0:
            exclusions["insufficient_history"] += 1
            continue
        expected_start = reference_open - (maximum_steps - 1) * CANDLE_INTERVAL
        if selected[start_index].open_time != expected_start:
            exclusions["missing_historical_candle"] += 1
            continue
        valid = True
        for column, definition in enumerate(VARIABLE_REGISTRY):
            count = int(definition.lookback / CANDLE_INTERVAL)
            first = reference_index - count + 1
            value = _calculate(
                definition,
                boundary=current_boundary,
                opens=opens[first : reference_index + 1],
                highs=highs[first : reference_index + 1],
                lows=lows[first : reference_index + 1],
                closes=closes[first : reference_index + 1],
                volumes=volumes[first : reference_index + 1],
            )
            if not math.isfinite(value):
                valid = False
                break
            feature_matrix[row, column] = np.float32(value)
        if not valid:
            feature_matrix[row, :] = np.nan
            exclusions["non_finite_feature"] += 1
            continue
        feature_eligible[row] = True
    boundary_tuple = tuple(boundaries)
    boundary_ns = np.asarray(
        [int(item.timestamp() * NANOSECONDS_PER_SECOND) for item in boundary_tuple],
        dtype=np.int64,
    )
    folds = tuple(
        TemporalFoldManifest(
            definition,
            np.flatnonzero(
                (
                    boundary_ns
                    >= int(
                        definition.training_start.timestamp() * NANOSECONDS_PER_SECOND
                    )
                )
                & (
                    boundary_ns
                    < int(definition.training_end.timestamp() * NANOSECONDS_PER_SECOND)
                )
            ).astype(np.int64),
            np.flatnonzero(
                (
                    boundary_ns
                    >= int(
                        definition.validation_start.timestamp() * NANOSECONDS_PER_SECOND
                    )
                )
                & (
                    boundary_ns
                    < int(
                        definition.validation_end.timestamp() * NANOSECONDS_PER_SECOND
                    )
                )
            ).astype(np.int64),
        )
        for definition in temporal_fold_definitions()
    )
    final_indices = np.flatnonzero(
        (boundary_ns >= int(FINAL_TEST_START.timestamp() * NANOSECONDS_PER_SECOND))
        & (boundary_ns < int(FINAL_TEST_END.timestamp() * NANOSECONDS_PER_SECOND))
    ).astype(np.int64)
    gaps = sum(
        1
        for left, right in pairwise(selected)
        if right.open_time - left.open_time != CANDLE_INTERVAL
    )
    metadata = MLDatasetMetadata(
        instrument,
        DEVELOPMENT_START,
        FINAL_TEST_END,
        selected[0].open_time if selected else None,
        selected[-1].open_time if selected else None,
        len(selected),
        gaps,
        len(VARIABLE_REGISTRY),
        MAXIMUM_HISTORICAL_LOOKBACK,
        _fingerprint(instrument, selected),
    )
    return MLExperimentDataset(
        metadata,
        boundary_tuple,
        feature_matrix,
        feature_eligible,
        ExclusionCounts.from_counter(exclusions),
        folds,
        final_indices,
        LazyTargetProvider(
            boundaries=boundary_tuple,
            reference_indices=reference_indices,
            opens=tuple(item.open_time for item in selected),
            closes=closes,
            feature_eligible=feature_eligible,
        ),
    )
