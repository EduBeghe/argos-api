from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.predictive_edge_engine_v4 import (
    _action_aligned_gross,
    _bounded_frozen_configurations,
    analyze_predictive_edges_v4,
    decode_v4_cache_value,
    encode_v4_cache_value,
)
from app.optimization.predictive_edge_v4 import EvidenceProfile, NeutralBandPolicy
from app.optimization.predictive_edges import Condition, SuggestedAction
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration


def _matrix(
    *, test_shift: Decimal = Decimal(0), row_count: int = 120
) -> HistoricalFeatureMatrix:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    price = Decimal("100")
    for index in range(row_count):
        at = start + timedelta(minutes=30 * index)
        price += Decimal("1") if index % 3 else Decimal("-0.25")
        visible_price = price + (test_shift if index >= 48 else Decimal(0))
        rows.append(
            HistoricalFeatureRow(
                at,
                visible_price,
                visible_price,
                visible_price - Decimal(1),
                Decimal(55),
                Decimal(2),
                Decimal("0.001") if index % 2 else Decimal("-0.001"),
                Decimal("0.002") if index % 3 else Decimal("-0.001"),
                Decimal("1.2"),
            )
        )
    return HistoricalFeatureMatrix("v4-dataset", tuple(rows))


def _window() -> WalkForwardWindow:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return WalkForwardWindow(
        1,
        start,
        start + timedelta(hours=12),
        start + timedelta(hours=12),
        start + timedelta(hours=24),
        start + timedelta(hours=24),
        start + timedelta(hours=48),
        48,
        "dataset",
        "selection",
    )


def _second_window() -> WalkForwardWindow:
    start = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=12)
    return WalkForwardWindow(
        2,
        start,
        start + timedelta(hours=12),
        start + timedelta(hours=12),
        start + timedelta(hours=24),
        start + timedelta(hours=24),
        start + timedelta(hours=48),
        48,
        "dataset-extended",
        "selection-extended",
    )


def test_real_v4_funnel_consumes_all_bounded_dimensions() -> None:
    result = analyze_predictive_edges_v4(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
    )
    assert result.training_counters.retained == 120
    assert result.validation_counters.generated == 40
    assert result.test_counters.generated <= 12
    assert len({item.horizon for item in result.cheap_relationships}) > 7
    assert len({item.cadence for item in result.cheap_relationships}) > 1
    cadence_observations: dict[tuple[str, timedelta], set[int]] = {}
    for item in result.cheap_relationships:
        cadence_observations.setdefault((item.predictor.name, item.horizon), set()).add(
            item.observation_count
        )
    assert any(len(values) > 1 for values in cadence_observations.values())
    assert {item.configuration.bins for item in result.validation_configurations} == {
        3,
        5,
        7,
    }
    assert {
        item.configuration.neutral_policy for item in result.validation_configurations
    } == set(NeutralBandPolicy)
    assert {
        item.configuration.profile for item in result.validation_configurations
    } == set(EvidenceProfile)
    assert {
        item.configuration.condition for item in result.validation_configurations
    } == set(Condition)
    assert all(
        len(item.configuration.bin_bounds) in (0, item.configuration.bins - 1)
        for item in result.validation_configurations
    )
    thresholds_by_policy = {
        policy: {
            item.configuration.neutral_threshold
            for item in result.validation_configurations
            if item.configuration.neutral_policy is policy
        }
        for policy in NeutralBandPolicy
    }
    assert thresholds_by_policy[NeutralBandPolicy.ECONOMIC_FRICTION] == {
        Decimal("0.0030")
    }
    assert thresholds_by_policy[NeutralBandPolicy.ZERO_DIRECTION] == {Decimal(0)}
    assert thresholds_by_policy[NeutralBandPolicy.TRAINING_NOISE] != {Decimal("0.0030")}
    assert any(item.benchmark is not None for item in result.validation_configurations)
    assert any(
        item.standard_error is not None for item in result.validation_configurations
    )


def test_action_aligned_economics_never_turns_adverse_movement_positive() -> None:
    round_trip_cost = Decimal("0.003")
    long_aligned = _action_aligned_gross(Decimal("-0.020"), SuggestedAction.LONG_ENTRY)
    exit_aligned = _action_aligned_gross(
        Decimal("0.020"), SuggestedAction.EXIT_OR_AVOID
    )
    assert long_aligned - round_trip_cost < 0
    assert exit_aligned - round_trip_cost < 0
    assert _action_aligned_gross(Decimal("1"), SuggestedAction.DIAGNOSTIC_ONLY) == 0


def test_training_ranked_variants_are_order_independent_and_explore_leaders() -> None:
    matrix = _matrix()
    windows = (_window(),)
    simulation = SimulationConfiguration()
    result = analyze_predictive_edges_v4(
        matrix=matrix, windows=windows, simulation=simulation
    )
    forward = _bounded_frozen_configurations(
        matrix, windows, simulation, result.cheap_relationships
    )
    reverse = _bounded_frozen_configurations(
        matrix, windows, simulation, tuple(reversed(result.cheap_relationships))
    )
    assert tuple(item.fingerprint for item in forward) == tuple(
        item.fingerprint for item in reverse
    )
    leading_relationship = max(
        result.cheap_relationships,
        key=lambda item: (item.score, item.observation_count, item.fingerprint),
    )
    leading_variants = [
        item
        for item in forward
        if item.predictor == leading_relationship.predictor
        and item.cadence == leading_relationship.cadence
        and item.horizon == leading_relationship.horizon
    ]
    assert len(leading_variants) > 1
    assert (
        len(
            {
                (item.bins, item.condition, item.neutral_policy, item.profile)
                for item in leading_variants
            }
        )
        > 1
    )
    assert (
        len({(item.predictor.name, item.cadence, item.horizon) for item in forward})
        >= 10
    )


def test_test_changes_cannot_change_validation_near_miss_ranking() -> None:
    base = analyze_predictive_edges_v4(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
    )
    changed = analyze_predictive_edges_v4(
        matrix=_matrix(test_shift=Decimal("10000")),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
    )
    assert tuple(item.diagnostic.fingerprint for item in base.near_misses) == tuple(
        item.diagnostic.fingerprint for item in changed.near_misses
    )


def test_stage_cache_reuses_real_work_and_force_recompute_bypasses_it() -> None:
    first = analyze_predictive_edges_v4(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
        cache_namespace="instrument-v4",
    )
    persisted = {
        key: decode_v4_cache_value(encode_v4_cache_value(value))
        for key, value in first.cache_writes.items()
    }
    second = analyze_predictive_edges_v4(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
        cache_namespace="instrument-v4",
        stage_cache=persisted,
    )
    forced = analyze_predictive_edges_v4(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
        cache_namespace="instrument-v4",
        stage_cache=persisted,
        force_recompute=True,
    )
    assert second.training_counters.computed == 0
    assert second.training_counters.reused == first.training_counters.generated
    assert second.validation_counters.computed == 0
    assert forced.training_counters.reused == 0
    assert forced.training_counters.computed == first.training_counters.generated


def test_extended_dataset_reuses_unchanged_slices_and_matches_forced_result() -> None:
    simulation = SimulationConfiguration()
    first = analyze_predictive_edges_v4(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=simulation,
        cache_namespace="stable-instrument-cost-contract",
    )
    persisted = {
        key: decode_v4_cache_value(encode_v4_cache_value(value))
        for key, value in first.cache_writes.items()
    }
    extended_matrix = _matrix(row_count=144)
    extended_windows = (_window(), _second_window())
    incremental = analyze_predictive_edges_v4(
        matrix=extended_matrix,
        windows=extended_windows,
        simulation=simulation,
        cache_namespace="stable-instrument-cost-contract",
        stage_cache=persisted,
    )
    forced = analyze_predictive_edges_v4(
        matrix=extended_matrix,
        windows=extended_windows,
        simulation=simulation,
        cache_namespace="stable-instrument-cost-contract",
        stage_cache=persisted,
        force_recompute=True,
    )
    assert incremental.training_counters.reused > 0
    assert incremental.training_counters.computed > 0
    assert (
        incremental.training_counters.computed < incremental.training_counters.generated
    )
    assert tuple(item.fingerprint for item in incremental.cheap_relationships) == tuple(
        item.fingerprint for item in forced.cheap_relationships
    )
    assert tuple(
        (item.configuration.fingerprint, item.status)
        for item in incremental.validation_configurations
    ) == tuple(
        (item.configuration.fingerprint, item.status)
        for item in forced.validation_configurations
    )


def test_weekly_research_runtime_imports_v4_not_legacy() -> None:
    from app.optimization import weekly_research

    assert (
        weekly_research.__dict__["analyze_predictive_edges_v4"]
        is analyze_predictive_edges_v4
    )
    assert not hasattr(weekly_research, "analyze_predictive_edges")
