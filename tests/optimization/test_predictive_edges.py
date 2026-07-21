from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.predictive_edges import (
    Condition,
    Direction,
    EvidenceStatus,
    EvidenceThresholds,
    PredictiveEdgeConfiguration,
    PredictiveEdgeError,
    _correlation,
    _metrics,
    _quantile_bounds,
    analyze_predictive_edges,
)
from app.optimization.walk_forward import WalkForwardConfiguration, WalkForwardWindow
from app.paper.config import SimulationConfiguration


def _row(at: datetime, close: str, return_1: str) -> HistoricalFeatureRow:
    value = Decimal(close)
    return HistoricalFeatureRow(
        at,
        value,
        value,
        value - Decimal("1"),
        Decimal("55"),
        Decimal("2"),
        Decimal(return_1),
        Decimal(return_1),
        Decimal("1.2"),
    )


def _matrix(*, mutate_after: datetime | None = None) -> HistoricalFeatureMatrix:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    price = Decimal("100")
    for index in range(120):
        at = start + timedelta(minutes=30 * index)
        price += Decimal("1") if index % 3 else Decimal("-0.25")
        if mutate_after is not None and at >= mutate_after:
            price += Decimal("1000")
        rows.append(_row(at, str(price), "0.001" if index % 2 else "-0.001"))
    return HistoricalFeatureMatrix("fingerprint", tuple(rows))


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


def _configuration() -> PredictiveEdgeConfiguration:
    return PredictiveEdgeConfiguration(
        horizons=(timedelta(minutes=30), timedelta(hours=1)),
        bins=5,
        conditions=(Condition.ALL,),
        windows=WalkForwardConfiguration(
            training_days=1,
            validation_days=1,
            test_days=1,
            step_days=1,
            maximum_windows=1,
            maximum_candidates=1,
            minimum_valid_windows=1,
        ),
        thresholds=EvidenceThresholds(
            minimum_observations=2,
            minimum_windows=1,
            minimum_stable_direction_ratio=Decimal("0.5"),
            minimum_absolute_correlation=Decimal("0"),
            minimum_directional_hit_rate=Decimal("0"),
        ),
    )


def test_configuration_rejects_unaligned_and_excessive_horizons() -> None:
    with pytest.raises(PredictiveEdgeError, match="align"):
        PredictiveEdgeConfiguration(horizons=(timedelta(minutes=31),))
    with pytest.raises(PredictiveEdgeError, match="14 days"):
        PredictiveEdgeConfiguration(horizons=(timedelta(days=15),))


def test_training_quantiles_are_deterministic_and_do_not_use_future_data() -> None:
    values = tuple(Decimal(value) for value in range(1, 101))
    assert _quantile_bounds(values, 5) == (
        Decimal("21"),
        Decimal("41"),
        Decimal("61"),
        Decimal("81"),
    )


def test_known_positive_negative_and_constant_correlations() -> None:
    values = tuple(Decimal(value) for value in range(5))
    assert _correlation(values, values) == Decimal(1)
    assert _correlation(values, tuple(reversed(values))) == Decimal(-1)
    assert _correlation((Decimal(1),) * 5, values) is None


def test_targets_are_exact_bounded_and_include_round_trip_friction() -> None:
    result = analyze_predictive_edges(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(
            fee_rate=Decimal("0.001"), slippage_rate=Decimal("0.0005")
        ),
        configuration=_configuration(),
    )
    edge = next(
        item
        for item in result.window_edges
        if item.predictor.name == "overall_score"
        and item.horizon == timedelta(minutes=30)
    )
    assert edge.metrics.usable_count == 47
    assert edge.metrics.missing_count == 1
    assert edge.metrics.mean_net_return < edge.metrics.average_absolute_movement
    assert edge.metrics.neutral_rate >= 0


def test_future_mutation_does_not_change_prior_window_evidence() -> None:
    boundary = datetime(2025, 1, 3, tzinfo=UTC)
    windows = (_window(),)
    simulation = SimulationConfiguration()
    configuration = _configuration()
    original = analyze_predictive_edges(
        matrix=_matrix(),
        windows=windows,
        simulation=simulation,
        configuration=configuration,
    )
    changed = analyze_predictive_edges(
        matrix=_matrix(mutate_after=boundary),
        windows=windows,
        simulation=simulation,
        configuration=configuration,
    )
    assert original.window_edges == changed.window_edges
    assert original.stability_edges == changed.stability_edges


def test_insufficient_observations_are_not_replaced_with_zero_correlation() -> None:
    metrics = _metrics((), 3, EvidenceThresholds())
    assert metrics.evidence_status is EvidenceStatus.INSUFFICIENT
    assert metrics.correlation is None
    assert metrics.insufficient_reason == "no usable observations"


def test_direction_vocabulary_is_explicit() -> None:
    assert tuple(Direction) == (Direction.UP, Direction.DOWN, Direction.NEUTRAL)

