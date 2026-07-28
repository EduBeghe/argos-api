from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.ml.contract import CANDLE_INTERVAL, DEVELOPMENT_START
from app.ml.dataset import MLDatasetError, build_ml_experiment_dataset
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


def test_boundary_uses_only_last_fully_closed_candle_and_target_is_exact() -> None:
    dataset = build_ml_experiment_dataset(instrument=INSTRUMENT, candles=candles())
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
    original = build_ml_experiment_dataset(instrument=INSTRUMENT, candles=source)
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
        instrument=INSTRUMENT, candles=tuple(changed_future)
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
        instrument=INSTRUMENT, candles=tuple(changed_history)
    )
    assert not np.array_equal(original.features[row], history_dataset.features[row])


def test_missing_history_and_future_are_excluded_without_interpolation() -> None:
    source = candles()
    original = build_ml_experiment_dataset(instrument=INSTRUMENT, candles=source)
    row = int(np.flatnonzero(original.feature_eligible)[0])
    boundary = original.boundaries[row]
    historical_gap = boundary - timedelta(hours=1)
    history_dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT,
        candles=tuple(item for item in source if item.open_time != historical_gap),
    )
    assert not history_dataset.feature_eligible[row]
    assert history_dataset.feature_exclusions.as_dict()["missing_historical_candle"] > 0
    future_gap = boundary + timedelta(minutes=10)
    future_dataset = build_ml_experiment_dataset(
        instrument=INSTRUMENT,
        candles=tuple(item for item in source if item.open_time != future_gap),
    )
    target = future_dataset.targets.for_duration(timedelta(minutes=30))
    assert not target.eligible[row]
    assert target.exclusions.as_dict()["missing_future_interval_candle"] > 0


def test_fingerprint_identity_and_data_are_deterministic() -> None:
    source = candles()
    first = build_ml_experiment_dataset(instrument=INSTRUMENT, candles=source)
    second = build_ml_experiment_dataset(instrument=INSTRUMENT, candles=source)
    assert first.metadata.fingerprint == second.metadata.fingerprint
    eth = Instrument(
        Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
    )
    assert (
        first.metadata.fingerprint
        != build_ml_experiment_dataset(
            instrument=eth, candles=source
        ).metadata.fingerprint
    )
    changed = list(source)
    changed[0] = candle(changed[0].open_time, Decimal("101"))
    assert (
        first.metadata.fingerprint
        != build_ml_experiment_dataset(
            instrument=INSTRUMENT, candles=tuple(changed)
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
    dataset = build_ml_experiment_dataset(instrument=INSTRUMENT, candles=tuple(source))
    assert np.isfinite(dataset.features[dataset.feature_eligible]).all()
    target = dataset.targets.for_duration(timedelta(hours=1))
    assert target.values.ndim == 1
    assert not hasattr(dataset.targets, "matrix")


def test_final_test_rows_are_outside_every_development_fold() -> None:
    dataset = build_ml_experiment_dataset(instrument=INSTRUMENT, candles=candles())
    final_rows = set(dataset.final_test_indices.tolist())
    assert final_rows
    for fold in dataset.folds:
        assert final_rows.isdisjoint(fold.training_indices.tolist())
        assert final_rows.isdisjoint(fold.validation_indices.tolist())


def test_invalid_timezone_is_rejected() -> None:
    source = list(candles(1))
    source[0] = candle(source[0].open_time.replace(tzinfo=None), Decimal("100"))
    try:
        build_ml_experiment_dataset(instrument=INSTRUMENT, candles=tuple(source))
    except MLDatasetError as error:
        assert "UTC" in str(error)
    else:
        raise AssertionError("naive timestamps must fail closed")
