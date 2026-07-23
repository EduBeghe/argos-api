from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.optimization.edge_guided import generate_edge_guided_candidates
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.predictive_edge_engine_v41 import (
    CONTEXT_BASE,
    CONTEXT_RELATIONSHIP_LIMIT,
    DIRECTIONAL_RELATIONSHIP_LIMIT,
    INTERACTION_GENERATION_LIMIT,
    INTERACTION_RETENTION_LIMIT,
    ISOLATED_TEST_LIMIT,
    VALIDATION_LIMIT,
    PredictorRole,
    _evaluate,
    _sampled_pairs,
    analyze_predictive_edges_v41,
    build_derived_feature_rows,
    enrich_feature_matrix,
)
from app.optimization.predictive_edges import SuggestedAction
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration


def _matrix(
    *,
    interaction: bool = True,
    test_multiplier: Decimal = Decimal(1),
    row_count: int = 370,
    test_start_index: int = 250,
) -> HistoricalFeatureMatrix:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    price = Decimal(100)
    rows: list[HistoricalFeatureRow] = []
    for index in range(row_count):
        signal = Decimal(1) if index % 2 == 0 else Decimal(-1)
        volume = (Decimal("0.5"), Decimal("1.0"), Decimal("2.0"))[index % 3]
        rows.append(
            HistoricalFeatureRow(
                start + timedelta(minutes=5 * index),
                price,
                price,
                price,
                Decimal(50) + signal * Decimal(10),
                Decimal(1),
                signal * Decimal("0.01"),
                signal * Decimal("0.02"),
                volume,
            )
        )
        if interaction and volume == Decimal("2.0"):
            movement = Decimal("0.012") if signal > 0 else Decimal("-0.004")
        elif interaction:
            movement = Decimal("-0.006") if signal > 0 else Decimal("0.006")
        else:
            movement = Decimal("0.001") if index % 4 < 2 else Decimal("-0.001")
        if index >= test_start_index:
            movement *= test_multiplier
        price *= Decimal(1) + movement
    return HistoricalFeatureMatrix("v41-fixture", tuple(rows))


def _window() -> WalkForwardWindow:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return WalkForwardWindow(
        1,
        start,
        start + timedelta(minutes=600),
        start + timedelta(minutes=600),
        start + timedelta(minutes=1200),
        start + timedelta(minutes=1200),
        start + timedelta(minutes=1800),
        120,
        "dataset",
        "selection",
    )


def _second_window() -> WalkForwardWindow:
    start = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=600)
    return WalkForwardWindow(
        2,
        start,
        start + timedelta(minutes=600),
        start + timedelta(minutes=600),
        start + timedelta(minutes=1200),
        start + timedelta(minutes=1200),
        start + timedelta(minutes=1800),
        120,
        "dataset-extended",
        "selection-extended",
    )


def _third_window() -> WalkForwardWindow:
    start = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=1200)
    return WalkForwardWindow(
        3,
        start,
        start + timedelta(minutes=600),
        start + timedelta(minutes=600),
        start + timedelta(minutes=1200),
        start + timedelta(minutes=1200),
        start + timedelta(minutes=1800),
        120,
        "dataset-third",
        "selection-third",
    )


def _windows() -> tuple[WalkForwardWindow, ...]:
    return (_window(), _second_window(), _third_window())


def _non_overlapping_windows() -> tuple[WalkForwardWindow, ...]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    phase = timedelta(minutes=600)
    return tuple(
        WalkForwardWindow(
            ordinal,
            start + phase * offset,
            start + phase * (offset + 1),
            start + phase * (offset + 1),
            start + phase * (offset + 2),
            start + phase * (offset + 2),
            start + phase * (offset + 3),
            120,
            f"non-overlap-{ordinal}",
            f"non-overlap-selection-{ordinal}",
        )
        for ordinal, offset in enumerate((0, 3, 6), 1)
    )


def _regime_matrix() -> HistoricalFeatureMatrix:
    """A 4h directional effect exists only in alternating high-volume regimes."""

    start = datetime(2025, 1, 1, tzinfo=UTC)
    count = 3000
    prices = tuple(
        Decimal(100)
        + Decimal(position if position <= 48 else 96 - position) / Decimal(4)
        for index in range(count)
        for position in (index % 96,)
    )
    rows: list[HistoricalFeatureRow] = []
    for index, price in enumerate(prices):
        future = prices[min(index + 48, count - 1)]
        direction = Decimal("0.01") if future > price else Decimal("-0.01")
        high_volume = (index // 6) % 2 == 0
        signal = direction if high_volume else -direction
        rows.append(
            HistoricalFeatureRow(
                start + timedelta(minutes=5 * index),
                price,
                price,
                price,
                Decimal(50) + signal * Decimal(1000),
                Decimal(1),
                signal,
                signal,
                Decimal(2) if high_volume else Decimal("0.5"),
            )
        )
    return HistoricalFeatureMatrix("v41-regime-fixture", tuple(rows))


def _long_windows() -> tuple[WalkForwardWindow, ...]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    phase = timedelta(minutes=5 * 576)
    return tuple(
        WalkForwardWindow(
            ordinal,
            start + phase * offset,
            start + phase * (offset + 1),
            start + phase * (offset + 1),
            start + phase * (offset + 2),
            start + phase * (offset + 2),
            start + phase * (offset + 3),
            576,
            f"dataset-{ordinal}",
            f"selection-{ordinal}",
        )
        for ordinal, offset in enumerate(range(3), 1)
    )


def test_derived_predictors_are_backward_looking_and_constant_breakout_is_safe() -> (
    None
):
    rows = build_derived_feature_rows(_matrix(interaction=False))
    assert rows
    assert all(value.is_finite() for row in rows for value in row.values.values())
    assert any("ema_20_slope_12" in row.values for row in rows)
    assert any("rolling_breakout_position_288" in row.values for row in rows)


def test_context_predictors_never_become_actionable_by_themselves() -> None:
    result = analyze_predictive_edges_v41(
        matrix=_matrix(row_count=730),
        windows=_windows(),
        simulation=SimulationConfiguration(),
    )
    assert all(
        item.directional_predictor not in CONTEXT_BASE for item in result.interactions
    )
    assert all(
        item.context_predictor in result.context_relationships
        for item in result.interactions
    )
    assert result.diagnostic_near_misses


def test_regime_conditioned_pattern_is_bounded_and_can_confirm() -> None:
    result = analyze_predictive_edges_v41(
        matrix=_regime_matrix(),
        windows=_long_windows(),
        simulation=SimulationConfiguration(),
    )
    assert result.stage_a1.retained <= DIRECTIONAL_RELATIONSHIP_LIMIT
    assert result.stage_a2.retained <= CONTEXT_RELATIONSHIP_LIMIT
    assert result.stage_a3.generated <= INTERACTION_GENERATION_LIMIT
    assert result.stage_a3.retained <= INTERACTION_RETENTION_LIMIT
    assert result.stage_b.generated <= VALIDATION_LIMIT
    assert result.stage_c.generated <= ISOLATED_TEST_LIMIT
    assert any(
        item.relationship.context_predictor == "relative_volume_20"
        for item in result.candidates
    )
    assert all(
        item.relationship.action
        in (SuggestedAction.LONG_ENTRY, SuggestedAction.EXIT_OR_AVOID)
        for item in result.candidates
    )
    assert result.confirmed_candidates
    assert result.hypotheses
    generated = generate_edge_guided_candidates(
        hypotheses=result.hypotheses,
        simulation=SimulationConfiguration(),
        maximum_candidates=60,
    )
    assert generated
    assert all(item.context_predictor_name is not None for item in generated)
    assert all(
        item.cadence == hypothesis.minimum_evaluation_interval
        for item in generated
        for hypothesis in result.hypotheses
        if item.source_hypothesis_id == hypothesis.hypothesis_id
    )


def test_no_interaction_control_does_not_confirm() -> None:
    result = analyze_predictive_edges_v41(
        matrix=_matrix(interaction=False),
        windows=_windows(),
        simulation=SimulationConfiguration(),
    )
    assert not result.confirmed_candidates


def test_test_changes_do_not_change_training_validation_selection() -> None:
    base = analyze_predictive_edges_v41(
        matrix=_matrix(row_count=1100, test_start_index=960),
        windows=_non_overlapping_windows(),
        simulation=SimulationConfiguration(),
    )
    changed = analyze_predictive_edges_v41(
        matrix=_matrix(
            test_multiplier=Decimal("-5"),
            row_count=1100,
            test_start_index=960,
        ),
        windows=_non_overlapping_windows(),
        simulation=SimulationConfiguration(),
    )
    assert tuple(item.fingerprint for item in base.interactions) == tuple(
        item.fingerprint for item in changed.interactions
    )
    assert tuple(
        (
            item.selected_bin,
            item.directional_lower_bound,
            item.directional_upper_bound,
            item.direction_mapping,
            item.action,
            item.cadence,
            item.horizon,
            item.frozen_regime.lower_bound,
            item.frozen_regime.upper_bound,
        )
        for item in base.interactions
    ) == tuple(
        (
            item.selected_bin,
            item.directional_lower_bound,
            item.directional_upper_bound,
            item.direction_mapping,
            item.action,
            item.cadence,
            item.horizon,
            item.frozen_regime.lower_bound,
            item.frozen_regime.upper_bound,
        )
        for item in changed.interactions
    )
    assert tuple(
        item.relationship.fingerprint for item in base.actionable_near_misses
    ) == tuple(item.relationship.fingerprint for item in changed.actionable_near_misses)


def test_exact_cache_reuse_and_force_recompute() -> None:
    first = analyze_predictive_edges_v41(
        matrix=_matrix(row_count=730),
        windows=_windows(),
        simulation=SimulationConfiguration(),
    )
    second = analyze_predictive_edges_v41(
        matrix=_matrix(row_count=730),
        windows=_windows(),
        simulation=SimulationConfiguration(),
        stage_cache=first.cache_writes,
    )
    forced = analyze_predictive_edges_v41(
        matrix=_matrix(row_count=730),
        windows=_windows(),
        simulation=SimulationConfiguration(),
        stage_cache=first.cache_writes,
        force_recompute=True,
    )
    assert second.stage_a3.computed == 0
    assert second.stage_a1.computed == 0
    assert second.stage_a2.computed == 0
    assert second.stage_a3.reused > 0
    assert forced.stage_a3.reused == 0
    assert forced.stage_a3.computed > 0


def test_extended_dataset_reuses_unchanged_slices_and_matches_forced() -> None:
    first = analyze_predictive_edges_v41(
        matrix=_matrix(row_count=730),
        windows=_windows(),
        simulation=SimulationConfiguration(),
    )
    matrix = _matrix(row_count=850)
    start = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=1800)
    fourth = WalkForwardWindow(
        4,
        start,
        start + timedelta(minutes=600),
        start + timedelta(minutes=600),
        start + timedelta(minutes=1200),
        start + timedelta(minutes=1200),
        start + timedelta(minutes=1800),
        120,
        "dataset-fourth",
        "selection-fourth",
    )
    windows = (*_windows(), fourth)
    incremental = analyze_predictive_edges_v41(
        matrix=matrix,
        windows=windows,
        simulation=SimulationConfiguration(),
        stage_cache=first.cache_writes,
    )
    forced = analyze_predictive_edges_v41(
        matrix=matrix,
        windows=windows,
        simulation=SimulationConfiguration(),
        stage_cache=first.cache_writes,
        force_recompute=True,
    )
    assert incremental.stage_a3.reused > 0
    assert incremental.stage_a3.computed > 0
    assert incremental.stage_a1.reused > 0
    assert incremental.stage_a1.computed > 0
    assert incremental.stage_a2.reused > 0
    assert incremental.stage_a2.computed > 0
    assert tuple(item.fingerprint for item in incremental.interactions) == tuple(
        item.fingerprint for item in forced.interactions
    )
    assert tuple(item.confirmed for item in incremental.candidates) == tuple(
        item.confirmed for item in forced.candidates
    )


def test_predictor_roles_are_explicit() -> None:
    assert {item.value for item in PredictorRole} == {"directional", "context"}


def test_minimum_three_windows_and_scale_invariant_screening() -> None:
    one_window = analyze_predictive_edges_v41(
        matrix=_matrix(), windows=(_window(),), simulation=SimulationConfiguration()
    )
    assert not one_window.confirmed_candidates

    from app.optimization.predictive_edge_engine_v41 import _correlation

    original = (
        (Decimal("1"), Decimal("-2")),
        (Decimal("2"), Decimal("1")),
        (Decimal("4"), Decimal("5")),
    )
    scaled = tuple((value * Decimal(100), outcome) for value, outcome in original)
    assert _correlation(original) == _correlation(scaled)


def test_enriched_matrix_preserves_derived_values_for_strategy_execution() -> None:
    enriched = enrich_feature_matrix(_matrix())
    assert any(row.research_values for row in enriched.rows)


def test_horizon_uses_t_plus_horizon_and_cadence_samples_boundaries() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    block = 48
    block_count = 24
    prices: list[Decimal] = []
    current = Decimal(100)
    for block_index in range(block_count):
        prices.extend((current,) * block)
        current *= Decimal("1.02") if block_index % 2 == 0 else Decimal("0.98")
    rows = tuple(
        HistoricalFeatureRow(
            start + timedelta(minutes=5 * index),
            price,
            price,
            price,
            Decimal(50),
            Decimal(1),
            Decimal(1) if (index // block) % 2 == 0 else Decimal(-1),
            Decimal(0),
            Decimal(1),
        )
        for index, price in enumerate(prices)
    )
    matrix = HistoricalFeatureMatrix("horizon", rows)
    values = {
        item.candle_open_time: item.values
        for item in build_derived_feature_rows(matrix)
    }
    end = start + timedelta(minutes=5 * len(rows))
    window = WalkForwardWindow(
        1, start, end, start, end, start, end, len(rows), "horizon", "horizon"
    )

    five_minutes = _evaluate(
        values,
        matrix,
        (window,),
        "return_1",
        None,
        "training",
        SimulationConfiguration(),
        SuggestedAction.LONG_ENTRY,
        timedelta(hours=4),
        timedelta(minutes=5),
        Decimal(0),
        None,
    )
    four_hours = _evaluate(
        values,
        matrix,
        (window,),
        "return_1",
        None,
        "training",
        SimulationConfiguration(),
        SuggestedAction.LONG_ENTRY,
        timedelta(hours=4),
        timedelta(hours=4),
        Decimal(0),
        None,
    )
    four_hour_samples = _sampled_pairs(
        values,
        matrix,
        window,
        "training",
        "return_1",
        timedelta(hours=4),
        timedelta(hours=4),
    )
    eight_hour_samples = _sampled_pairs(
        values,
        matrix,
        window,
        "training",
        "return_1",
        timedelta(hours=8),
        timedelta(hours=4),
    )

    assert not five_minutes.economic
    assert four_hours.economic
    assert four_hours.gross_movement is not None
    assert four_hours.gross_movement > 0
    assert len(four_hour_samples) > len(eight_hour_samples)
