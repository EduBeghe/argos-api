from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.predictive_edges import (
    PREDICTORS,
    BinSelection,
    Condition,
    Direction,
    DirectionMapping,
    EvidenceStatus,
    EvidenceThresholds,
    Observation,
    PredictiveEdgeConclusion,
    PredictiveEdgeConfiguration,
    PredictiveEdgeError,
    PredictorSemanticKind,
    SuggestedAction,
    ValidationStatus,
    _bin_selections,
    _correlation,
    _fingerprint,
    _hypothesis,
    _metrics,
    _quantile_bounds,
    _select_for_test,
    _stability,
    analyze_predictive_edges,
)
from app.optimization.predictive_edges import (
    TestConfirmationStatus as ConfirmationStatus,
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
    assert edge.test_metrics.usable_count == 47
    assert edge.test_metrics.missing_count == 1
    assert (
        edge.test_metrics.mean_net_movement_magnitude_after_friction
        < edge.test_metrics.mean_absolute_movement
    )
    assert edge.test_metrics.neutral_rate >= 0


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
    metrics = _metrics(
        (),
        3,
        predictor=next(item for item in PREDICTORS if item.name == "overall_score"),
        center=Decimal(0),
        mapping=DirectionMapping.POSITIVE,
        friction=Decimal("0.003"),
        thresholds=EvidenceThresholds(),
    )
    assert metrics.evidence_status is EvidenceStatus.INSUFFICIENT
    assert metrics.correlation is None
    assert metrics.insufficient_reason == "no usable observations"


def test_direction_vocabulary_is_explicit() -> None:
    assert tuple(Direction) == (Direction.UP, Direction.DOWN, Direction.NEUTRAL)


def _observation(
    index: int, predictor: str, gross: str, *, friction: Decimal = Decimal("0.003")
) -> Observation:
    return Observation(
        datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=30 * index),
        Decimal(predictor),
        Decimal(gross),
        abs(Decimal(gross)),
        max(abs(Decimal(gross)) - friction, Decimal(0)),
        Direction.UP if Decimal(gross) > 0 else Direction.DOWN,
    )


def test_training_center_prevents_positive_only_rsi_from_becoming_always_up() -> None:
    predictor = next(item for item in PREDICTORS if item.name == "rsi_14")
    values = tuple(
        _observation(index, str(value), gross)
        for index, (value, gross) in enumerate(
            ((40, "-0.02"), (45, "-0.01"), (55, "0.01"), (60, "0.02"))
        )
    )
    metrics = _metrics(
        values,
        0,
        predictor=predictor,
        center=Decimal(50),
        mapping=DirectionMapping.POSITIVE,
        friction=Decimal("0.003"),
        thresholds=EvidenceThresholds(minimum_observations=2),
    )
    assert metrics.directional_hit_rate == Decimal(1)
    assert metrics.up_precision == Decimal(1)
    assert metrics.down_precision == Decimal(1)
    assert metrics.direction_benchmarks is not None
    assert metrics.direction_benchmarks.always_up == Decimal("0.5")


@pytest.mark.parametrize("name", ("atr_14_relative", "relative_volume_20"))
def test_magnitude_predictors_do_not_fabricate_directional_accuracy(name: str) -> None:
    predictor = next(item for item in PREDICTORS if item.name == name)
    assert predictor.semantic_kind is PredictorSemanticKind.MAGNITUDE_OR_ACTIVITY
    metrics = _metrics(
        (
            _observation(0, "1", "0.01"),
            _observation(1, "2", "-0.03"),
            _observation(2, "3", "0.05"),
        ),
        0,
        predictor=predictor,
        center=Decimal(0),
        mapping=DirectionMapping.NOT_APPLICABLE,
        friction=Decimal("0.003"),
        thresholds=EvidenceThresholds(minimum_observations=2),
    )
    assert metrics.correlation is not None
    assert metrics.directional_hit_rate is None
    assert metrics.direction_benchmarks is None
    assert metrics.mean_strategy_aligned_net_return is None


def test_strategy_return_movement_and_avoided_downside_are_distinct() -> None:
    predictor = next(item for item in PREDICTORS if item.name == "return_1")
    metrics = _metrics(
        (
            _observation(0, "1", "0.010"),
            _observation(1, "-1", "-0.100"),
        ),
        0,
        predictor=predictor,
        center=Decimal(0),
        mapping=DirectionMapping.POSITIVE,
        friction=Decimal("0.003"),
        thresholds=EvidenceThresholds(minimum_observations=2),
    )
    assert metrics.mean_strategy_aligned_gross_return == Decimal("0.010")
    assert metrics.mean_strategy_aligned_net_return == Decimal("0.007")
    assert metrics.mean_expected_avoided_downside == Decimal("0.100")
    assert metrics.mean_forward_gross_return == Decimal("-0.045")
    assert metrics.mean_net_movement_magnitude_after_friction == Decimal("0.052")


def test_validation_selection_is_required_before_test_confirmation() -> None:
    predictor = next(item for item in PREDICTORS if item.name == "return_1")
    values = tuple(
        _observation(
            index, "1" if index % 2 else "-1", "0.02" if index % 2 else "-0.02"
        )
        for index in range(20)
    )
    thresholds = EvidenceThresholds(
        minimum_observations=10,
        minimum_absolute_correlation=Decimal("0.1"),
        minimum_directional_hit_rate=Decimal("0.6"),
        minimum_hit_rate_improvement_over_benchmark=Decimal("0.1"),
    )
    metrics = _metrics(
        values,
        0,
        predictor=predictor,
        center=Decimal(0),
        mapping=DirectionMapping.POSITIVE,
        friction=Decimal("0.003"),
        thresholds=thresholds,
    )
    status, _ = _select_for_test(predictor, metrics, thresholds)
    assert status is ValidationStatus.SELECTED
    contradictory = tuple(
        _observation(
            index, "1" if index % 2 else "-1", "-0.02" if index % 2 else "0.02"
        )
        for index in range(20)
    )
    contradiction_metrics = _metrics(
        contradictory,
        0,
        predictor=predictor,
        center=Decimal(0),
        mapping=DirectionMapping.POSITIVE,
        friction=Decimal("0.003"),
        thresholds=thresholds,
    )
    assert contradiction_metrics.mean_strategy_aligned_net_return is not None
    assert contradiction_metrics.mean_strategy_aligned_net_return < 0


def test_insufficient_history_is_explicit_and_has_no_hypotheses() -> None:
    configuration = PredictiveEdgeConfiguration(
        horizons=(timedelta(minutes=30),),
        thresholds=EvidenceThresholds(minimum_windows=2),
    )
    result = analyze_predictive_edges(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
        configuration=configuration,
    )
    assert result.conclusion is PredictiveEdgeConclusion.INSUFFICIENT_HISTORY
    assert result.windows_available == 1
    assert result.windows_required == 2
    assert result.hypotheses == ()


def _bin_observations(
    *, low_return: str, high_return: str, count: int = 10
) -> tuple[Observation, ...]:
    return tuple(
        _observation(index, "-2", low_return) for index in range(count)
    ) + tuple(_observation(index + count, "2", high_return) for index in range(count))


def test_negative_mapping_assigns_actions_per_bin_not_from_correlation_sign() -> None:
    selections = _bin_selections(
        validation=_bin_observations(low_return="0.02", high_return="-0.02"),
        test=_bin_observations(low_return="0.02", high_return="-0.02"),
        bounds=(Decimal("0"),),
        center=Decimal(0),
        mapping=DirectionMapping.NEGATIVE,
        friction=Decimal("0.003"),
        thresholds=EvidenceThresholds(
            minimum_observations=10,
            minimum_directional_hit_rate=Decimal("0.6"),
        ),
        enabled=True,
    )
    assert selections[0].suggested_action is SuggestedAction.LONG_ENTRY
    assert selections[0].test_confirmation_status is ConfirmationStatus.CONFIRMED
    assert selections[1].suggested_action is SuggestedAction.EXIT_OR_AVOID
    assert selections[1].test_confirmation_status is ConfirmationStatus.CONFIRMED
    assert selections[0].ordinal != selections[1].ordinal


def test_test_confirmation_cannot_leak_from_a_different_bin() -> None:
    validation = tuple(_observation(index, "-3", "0.02") for index in range(10))
    test = tuple(_observation(index, "-3", "-0.02") for index in range(10)) + tuple(
        _observation(index + 10, "3", "-0.02") for index in range(10)
    )
    selections = _bin_selections(
        validation=validation,
        test=test,
        bounds=(Decimal("-2"), Decimal("-1"), Decimal("0"), Decimal("1")),
        center=Decimal(0),
        mapping=DirectionMapping.NEGATIVE,
        friction=Decimal("0.003"),
        thresholds=EvidenceThresholds(
            minimum_observations=10,
            minimum_directional_hit_rate=Decimal("0.6"),
        ),
        enabled=True,
    )
    assert selections[0].selected_for_test is True
    assert selections[0].test_confirmation_status is ConfirmationStatus.REJECTED
    assert selections[4].selected_for_test is False
    assert selections[4].test_confirmation_status is ConfirmationStatus.NOT_SELECTED


def test_diagnostic_predictors_never_imply_short_or_directional_action() -> None:
    selections = _bin_selections(
        validation=_bin_observations(low_return="0.02", high_return="-0.02"),
        test=_bin_observations(low_return="0.02", high_return="-0.02"),
        bounds=(Decimal("0"),),
        center=Decimal(0),
        mapping=DirectionMapping.NOT_APPLICABLE,
        friction=Decimal("0.003"),
        thresholds=EvidenceThresholds(minimum_observations=10),
        enabled=True,
    )
    assert {item.suggested_action for item in selections} == {
        SuggestedAction.DIAGNOSTIC_ONLY
    }
    assert not any(item.selected_for_test for item in selections)
    assert "short" not in {item.value for item in SuggestedAction}


def test_bin_and_action_are_part_of_the_evidence_fingerprint() -> None:
    base = ("return_1", 3, 1800, "all", "negative")
    long_low = _fingerprint((*base, 1, SuggestedAction.LONG_ENTRY.value))
    exit_high = _fingerprint((*base, 5, SuggestedAction.EXIT_OR_AVOID.value))
    exit_low = _fingerprint((*base, 1, SuggestedAction.EXIT_OR_AVOID.value))
    assert len({long_low, exit_high, exit_low}) == 3


def test_two_bins_create_independent_action_hypotheses() -> None:
    configuration = _configuration()
    result = analyze_predictive_edges(
        matrix=_matrix(),
        windows=(_window(),),
        simulation=SimulationConfiguration(),
        configuration=configuration,
    )
    original = next(
        item for item in result.window_edges if item.predictor.name == "return_1"
    )

    def selection(ordinal: int, action: SuggestedAction) -> BinSelection:
        return BinSelection(
            ordinal,
            Decimal("-1") if ordinal == 1 else Decimal("1"),
            Decimal("-0.5") if ordinal == 1 else Decimal("2"),
            action,
            10,
            Decimal("0.02")
            if action is SuggestedAction.LONG_ENTRY
            else Decimal("-0.02"),
            Decimal("0.017") if action is SuggestedAction.LONG_ENTRY else None,
            Decimal("0.017") if action is SuggestedAction.EXIT_OR_AVOID else None,
            Decimal("0.8"),
            True,
            10,
            ConfirmationStatus.CONFIRMED,
            Decimal("0.02")
            if action is SuggestedAction.LONG_ENTRY
            else Decimal("-0.02"),
            Decimal("0.017") if action is SuggestedAction.LONG_ENTRY else None,
            Decimal("0.017") if action is SuggestedAction.EXIT_OR_AVOID else None,
            Decimal("0.8"),
            None,
        )

    edge = replace(
        original,
        direction_mapping=DirectionMapping.NEGATIVE,
        selected_for_test=True,
        selected_bin_ordinals=(1, 5),
        bin_selections=(
            selection(1, SuggestedAction.LONG_ENTRY),
            selection(5, SuggestedAction.EXIT_OR_AVOID),
        ),
        test_confirmation_status=ConfirmationStatus.CONFIRMED,
    )
    stability = _stability(
        (edge,),
        EvidenceThresholds(
            minimum_windows=1, minimum_stable_direction_ratio=Decimal("1")
        ),
    )
    hypotheses = tuple(
        _hypothesis(item, (edge.window_ordinal,), timedelta(minutes=30))
        for item in stability
    )
    assert {(item.bin_ordinal, item.suggested_action) for item in hypotheses} == {
        (1, SuggestedAction.LONG_ENTRY),
        (5, SuggestedAction.EXIT_OR_AVOID),
    }
    assert len({item.hypothesis_id for item in hypotheses}) == 2
