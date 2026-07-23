from datetime import UTC, datetime, timedelta
from decimal import Decimal

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
    analyze_predictive_edges_v41,
    build_derived_feature_rows,
)
from app.optimization.predictive_edges import SuggestedAction
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration


def _matrix(
    *,
    interaction: bool = True,
    test_multiplier: Decimal = Decimal(1),
    row_count: int = 370,
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
        if index >= 250:
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
        matrix=_matrix(), windows=(_window(),), simulation=SimulationConfiguration()
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
        matrix=_matrix(), windows=(_window(),), simulation=SimulationConfiguration()
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


def test_no_interaction_control_does_not_confirm() -> None:
    result = analyze_predictive_edges_v41(
        matrix=_matrix(interaction=False),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
    )
    assert not result.confirmed_candidates


def test_test_changes_do_not_change_training_validation_selection() -> None:
    base = analyze_predictive_edges_v41(
        matrix=_matrix(), windows=(_window(),), simulation=SimulationConfiguration()
    )
    changed = analyze_predictive_edges_v41(
        matrix=_matrix(test_multiplier=Decimal("-5")),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
    )
    assert tuple(item.fingerprint for item in base.interactions) == tuple(
        item.fingerprint for item in changed.interactions
    )
    assert tuple(
        item.relationship.fingerprint for item in base.actionable_near_misses
    ) == tuple(item.relationship.fingerprint for item in changed.actionable_near_misses)


def test_exact_cache_reuse_and_force_recompute() -> None:
    first = analyze_predictive_edges_v41(
        matrix=_matrix(), windows=(_window(),), simulation=SimulationConfiguration()
    )
    second = analyze_predictive_edges_v41(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
        stage_cache=first.cache_writes,
    )
    forced = analyze_predictive_edges_v41(
        matrix=_matrix(),
        windows=(_window(),),
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
        matrix=_matrix(), windows=(_window(),), simulation=SimulationConfiguration()
    )
    matrix = _matrix(row_count=490)
    windows = (_window(), _second_window())
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
    assert tuple(item.fingerprint for item in incremental.interactions) == tuple(
        item.fingerprint for item in forced.interactions
    )
    assert tuple(item.confirmed for item in incremental.candidates) == tuple(
        item.confirmed for item in forced.candidates
    )


def test_predictor_roles_are_explicit() -> None:
    assert {item.value for item in PredictorRole} == {"directional", "context"}
