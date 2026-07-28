from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.contract import CANDLE_INTERVAL, DEVELOPMENT_START
from app.ml.dataset import (
    ExclusionCounts,
    LazyTargetProvider,
    TargetVector,
    build_ml_experiment_dataset,
)
from app.ml.errors import MLDatasetError
from app.ml.reader import MLCandleColumns, candle_columns
from app.optimization.historical_data import HistoricalCandle

INSTRUMENT = Instrument(
    Asset.BTC_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def candle(open_time: datetime, close: Decimal) -> HistoricalCandle:
    return HistoricalCandle(
        open_time,
        open_time + CANDLE_INTERVAL - timedelta(milliseconds=1),
        close - Decimal("0.5"),
        close + Decimal("1"),
        close - Decimal("1"),
        close,
        Decimal("10"),
        Decimal("100"),
        10,
        Decimal("5"),
        Decimal("50"),
    )


def candles(days: int = 23) -> tuple[HistoricalCandle, ...]:
    count = days * 24 * 12
    return tuple(
        candle(DEVELOPMENT_START + index * CANDLE_INTERVAL, Decimal(100 + index) / 10)
        for index in range(count)
    )


def columns(source: tuple[HistoricalCandle, ...]) -> MLCandleColumns:
    return candle_columns(
        open_times=[item.open_time for item in source],
        close_times=[item.close_time for item in source],
        open_prices=[float(item.open_price) for item in source],
        high_prices=[float(item.high_price) for item in source],
        low_prices=[float(item.low_price) for item in source],
        close_prices=[float(item.close_price) for item in source],
        volumes=[float(item.base_asset_volume) for item in source],
    )


def test_boundary_uses_only_last_fully_closed_candle_and_target_is_exact() -> None:
    dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(candles())
    )
    eligible_rows = np.flatnonzero(dataset.feature_eligible)
    row = int(eligible_rows[0])
    boundary = dataset.boundaries[row]
    assert boundary.minute in {0, 30}
    target = dataset.targets.for_duration(timedelta(minutes=30))
    reference_index = int(
        (boundary - CANDLE_INTERVAL - DEVELOPMENT_START) / CANDLE_INTERVAL
    )
    future_index = int(
        (boundary + timedelta(minutes=25) - DEVELOPMENT_START) / CANDLE_INTERVAL
    )
    source = candles()
    expected = float(
        source[future_index].close_price / source[reference_index].close_price - 1
    )
    assert float(target.values[row]) == np.float32(expected)
    assert all(item.tzinfo is UTC for item in dataset.boundaries)
    assert dataset.feature_exclusions.total == len(dataset.boundaries) - int(
        dataset.feature_eligible.sum()
    )


def test_future_change_does_not_change_features_but_history_change_does() -> None:
    source = candles()
    original = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(source)
    )
    row = int(np.flatnonzero(original.feature_eligible)[0])
    boundary = original.boundaries[row]
    future_index = int(
        (boundary + timedelta(minutes=25) - DEVELOPMENT_START) / CANDLE_INTERVAL
    )
    changed_future = list(source)
    changed_future[future_index] = candle(
        changed_future[future_index].open_time, Decimal("999999")
    )
    future_dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(tuple(changed_future))
    )
    np.testing.assert_array_equal(original.features[row], future_dataset.features[row])
    historical_index = int(
        (boundary - CANDLE_INTERVAL - DEVELOPMENT_START) / CANDLE_INTERVAL
    )
    changed_history = list(source)
    changed_history[historical_index] = candle(
        changed_history[historical_index].open_time, Decimal("888888")
    )
    history_dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(tuple(changed_history))
    )
    assert not np.array_equal(original.features[row], history_dataset.features[row])


def test_missing_history_and_future_are_excluded_without_interpolation() -> None:
    source = candles()
    original = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(source)
    )
    row = int(np.flatnonzero(original.feature_eligible)[0])
    boundary = original.boundaries[row]
    historical_gap = boundary - timedelta(hours=1)
    history_dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT,
        columns=columns(
            tuple(item for item in source if item.open_time != historical_gap)
        ),
    )
    assert not history_dataset.feature_eligible[row]
    assert history_dataset.feature_exclusions.as_dict()["missing_historical_candle"] > 0
    future_gap = boundary + timedelta(minutes=10)
    future_dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT,
        columns=columns(tuple(item for item in source if item.open_time != future_gap)),
    )
    target = future_dataset.targets.for_duration(timedelta(minutes=30))
    assert not target.eligible[row]
    assert target.exclusions.as_dict()["missing_future_interval_candle"] > 0


def test_fingerprint_identity_and_data_are_deterministic() -> None:
    source = candles()
    first = build_ml_experiment_dataset(instrument=INSTRUMENT, columns=columns(source))
    second = build_ml_experiment_dataset(instrument=INSTRUMENT, columns=columns(source))
    assert first.metadata.fingerprint == second.metadata.fingerprint
    eth = Instrument(
        Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
    )
    assert (
        first.metadata.fingerprint
        != build_ml_experiment_dataset(
            instrument=eth, columns=columns(source)
        ).metadata.fingerprint
    )
    changed = list(source)
    changed[0] = candle(changed[0].open_time, Decimal("101"))
    assert (
        first.metadata.fingerprint
        != build_ml_experiment_dataset(
            instrument=INSTRUMENT, columns=columns(tuple(changed))
        ).metadata.fingerprint
    )


def test_zero_range_is_finite_and_targets_are_lazy_one_dimensional() -> None:
    source = list(candles())
    item = source[-1]
    source[-1] = HistoricalCandle(
        item.open_time,
        item.close_time,
        item.close_price,
        item.close_price,
        item.close_price,
        item.close_price,
        item.base_asset_volume,
        item.quote_asset_volume,
        item.trade_count,
        item.taker_buy_base_asset_volume,
        item.taker_buy_quote_asset_volume,
    )
    dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(tuple(source))
    )
    assert np.isfinite(dataset.features[dataset.feature_eligible]).all()
    target = dataset.targets.for_duration(timedelta(hours=1))
    assert target.values.ndim == 1
    assert not hasattr(dataset.targets, "matrix")


def test_final_test_rows_are_outside_every_development_fold() -> None:
    dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(candles())
    )
    final_rows = set(dataset.final_test_indices.tolist())
    assert final_rows
    for fold in dataset.folds:
        assert final_rows.isdisjoint(fold.training_indices.tolist())
        assert final_rows.isdisjoint(fold.validation_indices.tolist())


def test_safe_fold_indices_intersect_features_targets_and_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT, columns=columns(candles())
    )
    fold = dataset.folds[0]
    training_row = int(fold.training_indices[0])
    validation_row = int(fold.validation_indices[0])
    dataset.feature_eligible[[training_row, validation_row]] = True
    dataset.features[[training_row, validation_row]] = 0
    target_eligible = np.zeros(len(dataset.boundary_time_ns), dtype=np.bool_)
    target_eligible[[training_row, validation_row]] = True
    target_values = np.full(len(dataset.boundary_time_ns), np.nan, dtype=np.float32)
    target_values[[training_row, validation_row]] = np.float32("0.01")

    def target_for_duration(
        provider: LazyTargetProvider, duration: timedelta
    ) -> TargetVector:
        del provider
        return TargetVector(
            duration, target_values, target_eligible, ExclusionCounts(())
        )

    monkeypatch.setattr(LazyTargetProvider, "for_duration", target_for_duration)
    eligible = dataset.eligible_indices(
        fold_name=fold.definition.name, duration=timedelta(minutes=30)
    )
    assert eligible.training_indices.tolist() == [training_row]
    assert eligible.validation_indices.tolist() == [validation_row]
    assert np.isfinite(dataset.features[eligible.training_indices]).all()
    assert np.isfinite(target_values[eligible.validation_indices]).all()
    assert set(eligible.training_indices).isdisjoint(
        dataset.final_test_indices.tolist()
    )


def test_invalid_timezone_is_rejected() -> None:
    source = list(candles(1))
    source[0] = candle(source[0].open_time.replace(tzinfo=None), Decimal("100"))
    try:
        build_ml_experiment_dataset(
            instrument=INSTRUMENT, columns=columns(tuple(source))
        )
    except MLDatasetError as error:
        assert "UTC" in str(error)
    else:
        raise AssertionError("naive timestamps must fail closed")
