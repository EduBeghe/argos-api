from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.predictive_edge_engine_v4 import (
    analyze_predictive_edges_v4,
    decode_v4_cache_value,
    encode_v4_cache_value,
)
from app.optimization.predictive_edge_v4 import EvidenceProfile, NeutralBandPolicy
from app.optimization.predictive_edges import Condition
from app.optimization.walk_forward import WalkForwardWindow
from app.paper.config import SimulationConfiguration


def _matrix(*, test_shift: Decimal = Decimal(0)) -> HistoricalFeatureMatrix:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    price = Decimal("100")
    for index in range(120):
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


def test_weekly_research_runtime_imports_v4_not_legacy() -> None:
    from app.optimization import weekly_research

    assert (
        weekly_research.__dict__["analyze_predictive_edges_v4"]
        is analyze_predictive_edges_v4
    )
    assert not hasattr(weekly_research, "analyze_predictive_edges")
