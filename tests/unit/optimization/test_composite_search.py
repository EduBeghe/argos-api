from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.optimization.composite_search import (
    COMPOSITE_FEATURE_POOL,
    CompositeSearchBudget,
    CompositeSearchError,
    build_candidate,
    normalize_weights,
    search_composite_features,
)
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration


def _matrix(size: int = 420) -> HistoricalFeatureMatrix:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = tuple(
        HistoricalFeatureRow(
            candle_open_time=start + timedelta(minutes=5 * index),
            close_price=Decimal("100") + Decimal(index) / 10,
            ema_20=Decimal("99") + Decimal(index) / 10 + Decimal(index % 3) / 100,
            ema_50=Decimal("98") + Decimal(index) / 10 - Decimal(index % 5) / 100,
            rsi_14=Decimal(30 + index % 60),
            atr_14=Decimal("1") + Decimal(index % 7) / 100,
            return_1=Decimal(index % 11 - 5) / 1000,
            return_12=Decimal(index % 23 - 11) / 1000,
            relative_volume_20=Decimal("0.5") + Decimal(index % 17) / 10,
        )
        for index in range(size)
    )
    return HistoricalFeatureMatrix("dataset", rows)


def _window(matrix: HistoricalFeatureMatrix) -> tuple[WalkForwardWindow, ...]:
    rows = matrix.rows
    return (
        WalkForwardWindow(
            ordinal=0,
            training_from=rows[0].candle_open_time,
            training_to=rows[180].candle_open_time,
            validation_from=rows[180].candle_open_time,
            validation_to=rows[300].candle_open_time,
            test_from=rows[300].candle_open_time,
            test_to=rows[-1].candle_open_time + timedelta(minutes=5),
            decision_count=0,
            dataset_fingerprint=matrix.dataset_fingerprint,
            selection_fingerprint="selection",
        ),
    )


def _simulation() -> SimulationConfiguration:
    return SimulationConfiguration(1, "USDT", Decimal("0.001"), Decimal("0.0005"))


def test_weights_are_signed_unique_and_normalized() -> None:
    weights = normalize_weights((("rsi_14", Decimal("2")), ("return_1", Decimal("-1"))))
    assert tuple(item.feature_name for item in weights) == ("return_1", "rsi_14")
    assert sum(abs(item.weight) for item in weights) == Decimal(1)
    assert all(item.weight != 0 for item in weights)
    with pytest.raises(CompositeSearchError):
        normalize_weights((("rsi_14", Decimal(1)), ("rsi_14", Decimal(-1))))


def test_fingerprint_is_independent_of_input_order_and_training_only() -> None:
    matrix = _matrix()
    training = matrix.rows[:180]
    first = build_candidate(
        weights=(("rsi_14", Decimal(1)), ("return_1", Decimal(-1))),
        training_rows=training,
        simulation=_simulation(),
        cadence=timedelta(hours=4),
        horizon=timedelta(hours=12),
    )
    second = build_candidate(
        weights=(("return_1", Decimal(-1)), ("rsi_14", Decimal(1))),
        training_rows=training,
        simulation=_simulation(),
        cadence=timedelta(hours=4),
        horizon=timedelta(hours=12),
    )
    mutated_validation = tuple(
        replace(row, rsi_14=Decimal("999999")) for row in matrix.rows[180:]
    )
    third = build_candidate(
        weights=(("rsi_14", Decimal(1)), ("return_1", Decimal(-1))),
        training_rows=matrix.rows[:180] + mutated_validation[:0],
        simulation=_simulation(),
        cadence=timedelta(hours=4),
        horizon=timedelta(hours=12),
    )
    assert first == second == third


def test_progresses_deterministically_from_depth_two_through_seven() -> None:
    matrix = _matrix()
    budget = CompositeSearchBudget(
        beam_width_per_depth=4,
        maximum_variants_per_subset=1,
        maximum_cheap_evaluations=128,
        maximum_training_simulations=24,
        maximum_validation_per_depth=2,
        maximum_test_candidates=3,
        maximum_persisted_candidates=40,
    )
    first = search_composite_features(
        matrix=matrix, windows=_window(matrix), simulation=_simulation(), budget=budget
    )
    second = search_composite_features(
        matrix=matrix, windows=_window(matrix), simulation=_simulation(), budget=budget
    )
    assert tuple(item.depth for item in first.funnel_by_depth) == (2, 3, 4, 5, 6, 7)
    assert first.maximum_depth_reached == 7
    assert first == second
    assert first.total_evaluations <= budget.maximum_cheap_evaluations
    assert tuple(first.feature_pool) == COMPOSITE_FEATURE_POOL
    assert all(
        len(set(item.candidate.features)) == item.depth for item in first.candidates
    )


def test_constant_feature_is_excluded_with_reason() -> None:
    matrix = _matrix()
    constant = HistoricalFeatureMatrix(
        matrix.dataset_fingerprint,
        tuple(replace(row, rsi_14=Decimal(50)) for row in matrix.rows),
    )
    result = search_composite_features(
        matrix=constant, windows=_window(constant), simulation=_simulation()
    )
    assert result.excluded_features["rsi_14"] == "constant_feature"
