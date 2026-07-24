from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

import app.optimization.composite_search as composite_module
from app.market.catalog import MARKET_INSTRUMENTS
from app.optimization.composite_search import (
    COMPOSITE_FEATURE_POOL,
    CompositeCandidateStatus,
    CompositeMetrics,
    CompositeSearchBudget,
    CompositeSearchError,
    CompositeSelectionTrack,
    CompositeThresholdPolicy,
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


def _economic_matrix() -> tuple[HistoricalFeatureMatrix, tuple[WalkForwardWindow, ...]]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    price = Decimal("100")
    rows: list[HistoricalFeatureRow] = []
    size = 1440
    for index in range(size):
        common = Decimal(2) if (index // 120) % 2 == 0 else Decimal(-2)
        first_noise = Decimal((index * 37) % 101 - 50)
        second_noise = Decimal((index * 61 + 17) % 103 - 51)
        third_noise = -(first_noise + second_noise)
        first = common + first_noise
        second = common + second_noise
        third = common + third_noise
        aligned = common > 0
        rows.append(
            HistoricalFeatureRow(
                candle_open_time=start + timedelta(minutes=5 * index),
                close_price=price,
                ema_20=price * (Decimal(1) + Decimal(index % 7 - 3) / 10000),
                ema_50=price * (Decimal(1) + Decimal(index % 11 - 5) / 10000),
                rsi_14=Decimal(45 + index % 11),
                atr_14=price * (Decimal("0.005") + Decimal(index % 5) / 10000),
                return_1=first / Decimal(100),
                return_12=second / Decimal(100),
                relative_volume_20=Decimal(1) + third / Decimal(100),
            )
        )
        price *= Decimal("1.003") if aligned else Decimal("0.998")
    matrix = HistoricalFeatureMatrix("economic-dataset", tuple(rows))
    windows = tuple(
        WalkForwardWindow(
            ordinal=ordinal,
            training_from=rows[offset].candle_open_time,
            training_to=rows[offset + 240].candle_open_time,
            validation_from=rows[offset + 240].candle_open_time,
            validation_to=rows[offset + 360].candle_open_time,
            test_from=rows[offset + 360].candle_open_time,
            test_to=(
                rows[offset + 480].candle_open_time
                if offset + 480 < len(rows)
                else rows[-1].candle_open_time + timedelta(minutes=5)
            ),
            decision_count=0,
            dataset_fingerprint=matrix.dataset_fingerprint,
            selection_fingerprint=f"selection-{ordinal}",
        )
        for ordinal, offset in enumerate((0, 480, 960))
    )
    return matrix, windows


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
    assert tuple(item.depth for item in first.funnel_by_depth) == (2, 3, 4, 5, 6, 7)
    assert first.maximum_depth_reached == 7
    assert first == second
    assert first.cheap_evaluations_computed <= budget.maximum_cheap_evaluations
    assert first.total_evaluations == (
        first.cheap_evaluations_computed
        + first.training_simulations_computed
        + first.validation_simulations_computed
        + first.test_simulations_computed
    )
    assert tuple(first.feature_pool) == COMPOSITE_FEATURE_POOL
    assert all(
        len(set(item.candidate.features)) == item.depth for item in first.candidates
    )
    by_fingerprint = {item.candidate.fingerprint: item for item in first.candidates}
    deepest = next(item for item in first.candidates if item.depth == 7)
    lineage = [deepest]
    while lineage[-1].parent_fingerprint is not None:
        lineage.append(by_fingerprint[lineage[-1].parent_fingerprint])
    assert tuple(item.depth for item in lineage) == (7, 6, 5, 4, 3, 2)


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


def test_zero_economic_stability_never_uses_statistical_fallback() -> None:
    matrix = _matrix()
    candidate = build_candidate(
        weights=(("return_1", Decimal(1)), ("return_12", Decimal(1))),
        training_rows=matrix.rows[:180],
        simulation=_simulation(),
        cadence=timedelta(hours=4),
        horizon=timedelta(hours=4),
    )
    metrics = replace(
        _passing_metrics(),
        total_return=Decimal("-0.03836"),
        excess_return_over_production=Decimal("0.00676"),
        max_drawdown=Decimal("0.0801"),
        fees_paid=Decimal("2218"),
        fee_to_initial_cash_ratio=Decimal("0.2218"),
        return_stability=Decimal(0),
        profitable_window_ratio=Decimal(0),
        dispersion=Decimal("0.001"),
    )
    ranking = composite_module._ranking(candidate, metrics)
    assert ranking.stability == 0
    assert ranking.total < 0


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


def test_exploratory_parents_never_reach_validation_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _matrix()
    phases: list[str] = []

    def metrics(**kwargs: object) -> CompositeMetrics:
        phase = str(kwargs["phase"])
        phases.append(phase)
        return replace(
            _passing_metrics(),
            total_return=Decimal("-0.01"),
            return_stability=Decimal(0),
            profitable_window_ratio=Decimal(0),
        )

    monkeypatch.setattr(composite_module, "_economic_metrics", metrics)
    result = search_composite_features(
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=_window(matrix),
        simulation=_simulation(),
        initial_cash=Decimal("10000"),
        budget=CompositeSearchBudget(
            beam_width_per_depth=4,
            maximum_variants_per_subset=1,
            maximum_cheap_evaluations=80,
            maximum_training_simulations=24,
            maximum_validation_per_depth=2,
            maximum_test_candidates=2,
            maximum_persisted_candidates=40,
        ),
    )
    assert set(phases) == {"training"}
    assert result.maximum_depth_reached == 7
    assert all(
        item.selection_track is CompositeSelectionTrack.EXPLORATORY
        and item.status is CompositeCandidateStatus.RETAINED_TRAINING
        for item in result.candidates
    )


def test_churn_screen_rejects_noisy_30_minute_configuration() -> None:
    matrix = _matrix()
    candidate = build_candidate(
        weights=(("return_1", Decimal(1)), ("return_12", Decimal(1))),
        training_rows=matrix.rows[:180],
        simulation=_simulation(),
        cadence=timedelta(minutes=30),
        horizon=timedelta(minutes=30),
    )
    assert composite_module._excessive_churn(
        candidate,
        replace(_passing_metrics(), action_observation_ratio=Decimal("0.90")),
    )


def test_depth_quota_is_not_global_budget_exhaustion(
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
            beam_width_per_depth=4,
            maximum_variants_per_subset=1,
            maximum_cheap_evaluations=300,
            maximum_training_simulations=30,
            maximum_validation_per_depth=1,
            maximum_test_candidates=6,
            maximum_persisted_candidates=80,
        ),
    )
    assert any(
        item.stopped_reason
        is composite_module.CompositeDepthStoppedReason.DEPTH_QUOTA_REACHED
        for item in result.funnel_by_depth
    )
    assert result.maximum_depth_reached == 7
    assert result.budget_exhausted is False


def test_default_budget_reaches_every_depth_when_parents_are_eligible(
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
    )
    assert tuple(item.depth for item in result.funnel_by_depth) == (2, 3, 4, 5, 6, 7)
    assert result.maximum_depth_reached == 7
    assert all(item.cheap_evaluations_computed > 0 for item in result.funnel_by_depth)
    assert result.total_evaluations == (
        result.cheap_evaluations_computed
        + result.training_simulations_computed
        + result.validation_simulations_computed
        + result.test_simulations_computed
    )
    assert all(
        set(item.simulations_by_cadence or {}) == {"1800", "14400", "28800", "43200"}
        for item in result.funnel_by_depth
    )


def test_reduced_budget_stops_at_last_real_depth_without_empty_funnel(
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
            beam_width_per_depth=1,
            maximum_variants_per_subset=1,
            maximum_cheap_evaluations=2,
            maximum_training_simulations=1,
            maximum_validation_per_depth=1,
            maximum_test_candidates=1,
            maximum_persisted_candidates=4,
        ),
    )
    assert result.maximum_depth_reached == 2
    assert tuple(item.depth for item in result.funnel_by_depth) == (2,)
    assert result.funnel_by_depth[0].cheap_evaluations_computed == 1
    assert result.budget_exhausted is True


def test_validation_pass_without_test_slot_has_explicit_status(
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
            maximum_cheap_evaluations=24,
            maximum_training_simulations=12,
            maximum_validation_per_depth=2,
            maximum_test_candidates=1,
            maximum_persisted_candidates=20,
        ),
    )
    statuses = tuple(item.status for item in result.candidates)
    assert CompositeCandidateStatus.CONFIRMED_CANDIDATE in statuses
    assert CompositeCandidateStatus.PASSED_VALIDATION in statuses
    funnel = result.funnel_by_depth[0]
    assert funnel.validation_retained == 2
    assert funnel.test_evaluated == 1
    assert result.test_simulations_computed == 1
    assert (
        result.conclusion
        is composite_module.CompositeSearchConclusion.CONFIRMED_COMPOSITE_CANDIDATES
    )


def test_three_feature_economic_signal_reaches_training_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix, windows = _economic_matrix()
    monkeypatch.setattr(
        composite_module,
        "COMPOSITE_FEATURE_POOL",
        ("return_1", "return_12", "relative_volume_20"),
    )
    first_training = tuple(
        row
        for row in matrix.rows
        if windows[0].training_from <= row.candle_open_time < windows[0].training_to
    )
    assert all(
        abs(composite_module._association(first_training, name, timedelta(hours=4)))
        < Decimal("0.10")
        for name in ("return_1", "return_12", "relative_volume_20")
    )
    result = search_composite_features(
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=windows,
        simulation=SimulationConfiguration(
            1, "USDT", Decimal("0.0001"), Decimal("0.0001")
        ),
        initial_cash=Decimal("10000"),
        budget=CompositeSearchBudget(
            beam_width_per_depth=6,
            maximum_variants_per_subset=2,
            maximum_cheap_evaluations=180,
            maximum_training_simulations=36,
            maximum_validation_per_depth=6,
            maximum_test_candidates=6,
            maximum_persisted_candidates=60,
        ),
    )
    assert all(
        item.selection_track is CompositeSelectionTrack.EXPLORATORY
        for item in result.candidates
        if item.depth == 2
    )
    depth_three = tuple(
        item
        for item in result.candidates
        if item.depth == 3
        and {"return_1", "return_12", "relative_volume_20"}
        <= set(item.candidate.features)
    )
    assert depth_three
    by_fingerprint = {item.candidate.fingerprint: item for item in result.candidates}
    assert any(
        item.parent_fingerprint is not None
        and by_fingerprint[item.parent_fingerprint].selection_track
        is CompositeSelectionTrack.EXPLORATORY
        for item in depth_three
    )
    assert any(item.training_metrics.total_return > 0 for item in depth_three)
    assert any(
        item.status
        in {
            CompositeCandidateStatus.PASSED_VALIDATION,
            CompositeCandidateStatus.REJECTED_TEST,
            CompositeCandidateStatus.CONFIRMED_CANDIDATE,
        }
        for item in depth_three
    )


def test_quantile_thresholds_are_frozen_from_training_only() -> None:
    matrix = _matrix()
    training = matrix.rows[:180]
    candidate = build_candidate(
        weights=(("return_1", Decimal(1)), ("return_12", Decimal(1))),
        training_rows=training,
        simulation=_simulation(),
        cadence=timedelta(hours=4),
        horizon=timedelta(hours=4),
        threshold_policy=CompositeThresholdPolicy.TRAINING_QUANTILE,
        entry_quantile=Decimal("0.85"),
        exit_quantile=Decimal("0.45"),
    )
    mutated_held_out = tuple(
        replace(row, return_1=Decimal("9999"), return_12=Decimal("-9999"))
        for row in matrix.rows[180:]
    )
    repeated = build_candidate(
        weights=(("return_1", Decimal(1)), ("return_12", Decimal(1))),
        training_rows=training + mutated_held_out[:0],
        simulation=_simulation(),
        cadence=timedelta(hours=4),
        horizon=timedelta(hours=4),
        threshold_policy=CompositeThresholdPolicy.TRAINING_QUANTILE,
        entry_quantile=Decimal("0.85"),
        exit_quantile=Decimal("0.45"),
    )
    assert candidate.entry_threshold == repeated.entry_threshold
    assert candidate.exit_threshold == repeated.exit_threshold
    assert candidate.fingerprint == repeated.fingerprint


def test_real_costs_reject_the_same_gross_composite_signal() -> None:
    matrix, windows = _economic_matrix()
    result = search_composite_features(
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=windows,
        simulation=SimulationConfiguration(1, "USDT", Decimal("0.05"), Decimal("0.05")),
        initial_cash=Decimal("10000"),
        budget=CompositeSearchBudget(
            beam_width_per_depth=6,
            maximum_variants_per_subset=2,
            maximum_cheap_evaluations=180,
            maximum_training_simulations=36,
            maximum_validation_per_depth=6,
            maximum_test_candidates=6,
            maximum_persisted_candidates=60,
        ),
    )
    assert not any(
        item.status
        in {
            CompositeCandidateStatus.PASSED_VALIDATION,
            CompositeCandidateStatus.CONFIRMED_CANDIDATE,
        }
        for item in result.candidates
    )


def test_real_cadence_changes_decisions_and_trades() -> None:
    matrix, windows = _economic_matrix()
    price = Decimal("100")
    cadence_rows: list[HistoricalFeatureRow] = []
    for index, row in enumerate(matrix.rows):
        signal = Decimal(1) if (index // 12) % 2 == 0 else Decimal(-1)
        cadence_rows.append(
            replace(
                row,
                close_price=price,
                ema_20=price,
                ema_50=price,
                atr_14=price * Decimal("0.005"),
                return_1=signal / Decimal(50),
                return_12=signal / Decimal(50),
                relative_volume_20=Decimal(1) + signal / Decimal(50),
            )
        )
        price *= Decimal("1.002") if signal > 0 else Decimal("0.999")
    matrix = HistoricalFeatureMatrix(matrix.dataset_fingerprint, tuple(cadence_rows))
    simulation = SimulationConfiguration(
        1, "USDT", Decimal("0.0001"), Decimal("0.0001")
    )
    training_rows = composite_module._phase_rows(matrix, windows, "training")
    weights = (
        ("return_1", Decimal(1)),
        ("return_12", Decimal(1)),
        ("relative_volume_20", Decimal(1)),
    )
    thirty_minutes = build_candidate(
        weights=weights,
        training_rows=training_rows,
        simulation=simulation,
        cadence=timedelta(minutes=30),
        horizon=timedelta(minutes=30),
        entry_threshold=Decimal("0.03"),
        exit_threshold=Decimal("-0.03"),
        initial_cash=Decimal("10000"),
    )
    four_hours = build_candidate(
        weights=weights,
        training_rows=training_rows,
        simulation=simulation,
        cadence=timedelta(hours=4),
        horizon=timedelta(hours=4),
        entry_threshold=Decimal("0.03"),
        exit_threshold=Decimal("-0.03"),
        initial_cash=Decimal("10000"),
    )
    fast = composite_module._economic_metrics(
        candidate=thirty_minutes,
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=windows,
        phase="training",
        simulation=simulation,
        initial_cash=Decimal("10000"),
    )
    slow = composite_module._economic_metrics(
        candidate=four_hours,
        dataset=_dataset(matrix),
        matrix=matrix,
        windows=windows,
        phase="training",
        simulation=simulation,
        initial_cash=Decimal("10000"),
    )
    assert fast.observations > slow.observations
    assert fast.trade_count != slow.trade_count


def test_economic_evaluation_reuses_shared_benchmarks() -> None:
    matrix, windows = _economic_matrix()
    simulation = SimulationConfiguration(
        1, "USDT", Decimal("0.0001"), Decimal("0.0001")
    )
    training_rows = composite_module._phase_rows(matrix, windows, "training")
    candidates = tuple(
        build_candidate(
            weights=weights,
            training_rows=training_rows,
            simulation=simulation,
            cadence=timedelta(hours=4),
            horizon=timedelta(hours=4),
            threshold_policy=CompositeThresholdPolicy.TRAINING_QUANTILE,
            entry_quantile=Decimal("0.85"),
            exit_quantile=Decimal("0.45"),
        )
        for weights in (
            (("return_1", Decimal(1)), ("return_12", Decimal(1))),
            (("return_1", Decimal(1)), ("relative_volume_20", Decimal(1))),
        )
    )
    cache = composite_module._EvaluationCache()
    for candidate in candidates:
        composite_module._economic_metrics(
            candidate=candidate,
            dataset=_dataset(matrix),
            matrix=matrix,
            windows=windows,
            phase="training",
            simulation=simulation,
            initial_cash=Decimal("10000"),
            cache=cache,
        )
    assert cache.production_simulations_computed == len(windows)
    assert cache.production_simulations_reused == len(windows)
    assert cache.buy_and_hold_simulations_computed == len(windows)
    assert cache.buy_and_hold_simulations_reused == len(windows)


def test_validation_and_test_mutation_preserves_training_generation_order() -> None:
    matrix, windows = _economic_matrix()
    held_out_ranges = tuple(
        (window.validation_from, window.test_to) for window in windows
    )
    mutated = HistoricalFeatureMatrix(
        "mutated-held-out",
        tuple(
            replace(
                row,
                close_price=row.close_price * Decimal("100"),
                return_1=row.return_1 * Decimal("-50"),
                return_12=row.return_12 * Decimal("75"),
                relative_volume_20=row.relative_volume_20 * Decimal("10"),
            )
            if any(
                start <= row.candle_open_time < end for start, end in held_out_ranges
            )
            else row
            for row in matrix.rows
        ),
    )
    simulation = SimulationConfiguration(
        1, "USDT", Decimal("0.0001"), Decimal("0.0001")
    )
    budget = CompositeSearchBudget(
        beam_width_per_depth=4,
        maximum_variants_per_subset=2,
        maximum_cheap_evaluations=120,
        maximum_training_simulations=24,
        maximum_validation_per_depth=4,
        maximum_test_candidates=4,
        maximum_persisted_candidates=40,
    )

    def generated(value: HistoricalFeatureMatrix) -> tuple[object, ...]:
        result = search_composite_features(
            dataset=_dataset(value),
            matrix=value,
            windows=windows,
            simulation=simulation,
            initial_cash=Decimal("10000"),
            budget=budget,
        )
        return tuple(
            (
                item.depth,
                item.parent_fingerprint,
                item.candidate.fingerprint,
                item.candidate.weights,
                item.candidate.entry_threshold,
                item.candidate.exit_threshold,
                item.candidate.cadence,
                item.candidate.prediction_horizon,
                item.candidate.minimum_holding_period,
                item.candidate.maximum_holding_period,
                item.candidate.cooldown,
            )
            for item in result.candidates
        )

    assert generated(matrix) == generated(mutated)
