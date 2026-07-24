from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

import app.optimization.composite_search as composite_module
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.composite_search import (
    COMPOSITE_FEATURE_POOL,
    CompositeMetrics,
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
from app.optimization.historical_data import (
    HistoricalDataset,
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
)
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration
from app.strategies.baseline_long_only import StrategyAction


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


def _dataset(matrix: HistoricalFeatureMatrix) -> HistoricalDataset:
    return HistoricalDataset(
        HistoricalDatasetIdentity(
            MARKET_INSTRUMENTS[0].instrument,
            HistoricalDatasetRange(
                matrix.rows[0].candle_open_time,
                matrix.rows[-1].candle_open_time + timedelta(minutes=5),
            ),
        ),
        0,
        0,
        None,
        None,
        0,
        None,
        None,
        (),
        matrix.dataset_fingerprint,
    )


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
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=_window(matrix),
        simulation=_simulation(),
        initial_cash=Decimal("10000"),
        budget=budget,
    )
    second = search_composite_features(
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=_window(matrix),
        simulation=_simulation(),
        initial_cash=Decimal("10000"),
        budget=budget,
    )
    assert tuple(item.depth for item in first.funnel_by_depth) == (2,)
    assert first.maximum_depth_reached == 2
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
        dataset=_dataset(constant),
        matrix=constant,
        windows=_window(constant),
        simulation=_simulation(),
        initial_cash=Decimal("10000"),
    )
    assert result.excluded_features["rsi_14"] == "constant_feature"


def _passing_metrics() -> CompositeMetrics:
    return CompositeMetrics(
        observations=300,
        correlation=Decimal("0.4"),
        directional_hit_rate=Decimal("0.7"),
        expected_gross_movement=Decimal("0.03"),
        expected_net_movement=Decimal("0.02"),
        dispersion=Decimal("0.01"),
        profitable_window_ratio=Decimal("1"),
        usable_window_count=3,
        total_return=Decimal("0.05"),
        absolute_pnl=Decimal("500"),
        max_drawdown=Decimal("0.05"),
        trade_count=9,
        fees_paid=Decimal("12"),
        final_equity=Decimal("10500"),
        production_baseline_return=Decimal("0.01"),
        excess_return_over_production=Decimal("0.04"),
        buy_and_hold_return=Decimal("0.03"),
        excess_return_over_buy_and_hold=Decimal("0.02"),
        return_stability=Decimal("1"),
    )


def test_beam_children_add_exactly_one_feature_to_real_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    monkeypatch.setattr(
        composite_module, "_economic_metrics", lambda **_: _passing_metrics()
    )
    result = search_composite_features(
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=_window(matrix),
        simulation=_simulation(),
        initial_cash=Decimal("10000"),
        budget=CompositeSearchBudget(
            beam_width_per_depth=2,
            maximum_variants_per_subset=1,
            maximum_cheap_evaluations=300,
            maximum_training_simulations=20,
            maximum_validation_per_depth=1,
            maximum_test_candidates=6,
            maximum_persisted_candidates=80,
        ),
    )
    by_fingerprint = {item.candidate.fingerprint: item for item in result.candidates}
    children = tuple(item for item in result.candidates if item.depth > 2)
    assert children
    for child in children:
        assert child.parent_fingerprint in by_fingerprint
        parent = by_fingerprint[child.parent_fingerprint]
        assert set(parent.candidate.features) < set(child.candidate.features)
        assert len(child.candidate.features) == len(parent.candidate.features) + 1


def test_validation_and_test_mutation_cannot_change_training_candidate() -> None:
    matrix = _matrix()
    windows = _window(matrix)
    evidence = composite_module._training_evidence(matrix, windows, timedelta(hours=4))
    mutated = HistoricalFeatureMatrix(
        matrix.dataset_fingerprint,
        tuple(
            replace(row, close_price=row.close_price * Decimal("1000"))
            if row.candle_open_time >= windows[0].validation_from
            else row
            for row in matrix.rows
        ),
    )
    assert (
        composite_module._training_evidence(mutated, windows, timedelta(hours=4))
        == evidence
    )


def test_composite_rules_respect_entry_exit_holding_and_cooldown() -> None:
    matrix = _matrix()
    candidate = build_candidate(
        weights=(("return_1", Decimal(1)), ("return_12", Decimal(1))),
        training_rows=matrix.rows[:180],
        simulation=_simulation(),
        cadence=timedelta(minutes=30),
        horizon=timedelta(minutes=30),
        entry_threshold=Decimal("-100"),
        exit_threshold=Decimal("100"),
        minimum_holding_period=timedelta(hours=1),
        maximum_holding_period=timedelta(hours=2),
        cooldown=timedelta(hours=1),
    )
    decide = composite_module._composite_action_filter(candidate)
    assert (
        decide(matrix.rows[0], False, StrategyAction.NO_ACTION).action
        is StrategyAction.OPEN_POSITION
    )
    assert (
        decide(matrix.rows[1], True, StrategyAction.NO_ACTION)
        is StrategyAction.NO_ACTION
    )
    later = replace(
        matrix.rows[2],
        candle_open_time=matrix.rows[0].candle_open_time + timedelta(hours=2),
    )
    assert (
        decide(later, True, StrategyAction.NO_ACTION).action is StrategyAction.SELL_ALL
    )
    assert (
        decide(matrix.rows[3], False, StrategyAction.NO_ACTION)
        is StrategyAction.NO_ACTION
    )


def test_validation_rejection_does_not_consult_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    phases: list[str] = []

    def metrics(**kwargs: object) -> CompositeMetrics:
        phase = str(kwargs["phase"])
        phases.append(phase)
        if phase == "validation":
            return replace(
                _passing_metrics(),
                total_return=Decimal("-0.01"),
                expected_net_movement=Decimal("-0.01"),
            )
        return _passing_metrics()

    monkeypatch.setattr(composite_module, "_economic_metrics", metrics)
    result = search_composite_features(
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=_window(matrix),
        simulation=_simulation(),
        initial_cash=Decimal("10000"),
        budget=CompositeSearchBudget(
            beam_width_per_depth=1,
            maximum_variants_per_subset=1,
            maximum_cheap_evaluations=40,
            maximum_training_simulations=2,
            maximum_validation_per_depth=1,
            maximum_test_candidates=1,
            maximum_persisted_candidates=10,
        ),
    )
    assert "validation" in phases
    assert "test" not in phases
    assert all(item.test_metrics is None for item in result.candidates)
