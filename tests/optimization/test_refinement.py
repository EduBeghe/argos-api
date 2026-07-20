from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from app.market.catalog import Instrument
from app.market.enums import Asset, Interval, MarketType, Provider
from app.optimization.feature_matrix import (
    HistoricalFeatureMatrix,
    HistoricalFeatureRow,
)
from app.optimization.historical_data import (
    HistoricalDataset,
    HistoricalDatasetIdentity,
    HistoricalDatasetRange,
)
from app.optimization.refinement import (
    CandidateSegmentResult,
    HistoricalRefinementError,
    RecommendationStatus,
    RefinementSegmentName,
    _ranking_key,
    generate_baseline_long_only_candidates,
    refine_historical_dataset,
    refine_historical_strategy,
    split_feature_matrix,
)
from app.optimization.simulation import HistoricalSimulationResult
from app.strategies.baseline_long_only import (
    BASELINE_LONG_ONLY_V1_PARAMETERS,
    BaselineLongOnlyParameters,
)

START = datetime(2024, 1, 1, tzinfo=UTC)
INSTRUMENT = Instrument(
    Asset.ETH_USDT, Provider.BINANCE, MarketType.SPOT, Interval.FIVE_MINUTES
)


def dataset() -> HistoricalDataset:
    return HistoricalDataset(
        HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, START + timedelta(hours=6))
        ),
        72,
        72,
        START,
        START + timedelta(hours=6) - timedelta(minutes=5),
        0,
        None,
        None,
        (),
        "refinement-fingerprint",
    )


def matrix() -> HistoricalFeatureMatrix:
    rows = tuple(
        HistoricalFeatureRow(
            START + timedelta(minutes=25 + 30 * index),
            Decimal("100"),
            Decimal("1"),
            Decimal("1"),
            Decimal("50"),
            Decimal("1"),
            Decimal("0"),
            Decimal("0"),
            Decimal("1"),
        )
        for index in range(10)
    )
    return HistoricalFeatureMatrix("refinement-fingerprint", rows)


def fake_simulation(
    *,
    test_candidate_return: Decimal = Decimal("0.03"),
    validation_positive: bool = True,
    validation_trades: int = 2,
) -> tuple[
    list[tuple[RefinementSegmentName, object, Decimal, Decimal]],
    Callable[..., HistoricalSimulationResult],
]:
    calls: list[tuple[RefinementSegmentName, object, Decimal, Decimal]] = []

    def simulate(**kwargs: object) -> HistoricalSimulationResult:
        feature_matrix = cast(HistoricalFeatureMatrix, kwargs["feature_matrix"])
        parameters = cast(BaselineLongOnlyParameters, kwargs["strategy_parameters"])
        first_hour = feature_matrix.rows[0].candle_open_time.hour
        segment = (
            RefinementSegmentName.TRAINING
            if first_hour == 0
            else RefinementSegmentName.VALIDATION
            if first_hour == 3
            else RefinementSegmentName.TEST
        )
        fee_rate = cast(Decimal, kwargs["fee_rate"])
        slippage_rate = cast(Decimal, kwargs["slippage_rate"])
        calls.append((segment, parameters, fee_rate, slippage_rate))
        is_best = (
            parameters.entry_threshold == Decimal("0.70")
            and parameters.exit_threshold == Decimal("0.30")
            and parameters.open_position_target_percentage == Decimal("20")
        )
        is_baseline = parameters == BASELINE_LONG_ONLY_V1_PARAMETERS
        if segment is RefinementSegmentName.TRAINING:
            total_return = (
                parameters.entry_threshold
                + parameters.exit_threshold / Decimal(10)
                + parameters.open_position_target_percentage / Decimal(10000)
            )
        elif segment is RefinementSegmentName.VALIDATION:
            total_return = (
                Decimal("0.05") if validation_positive and is_best else Decimal("-0.01")
            )
        else:
            total_return = (
                Decimal("0.01")
                if is_baseline
                else test_candidate_return
                if is_best
                else Decimal("-0.02")
            )
        trade_count = (
            validation_trades if segment is RefinementSegmentName.VALIDATION else 3
        )
        return cast(
            HistoricalSimulationResult,
            SimpleNamespace(
                total_return=total_return,
                max_drawdown=Decimal("0.04"),
                trade_count=trade_count,
                fees_paid=Decimal("2"),
                final_equity=Decimal("10000") * (Decimal(1) + total_return),
            ),
        )

    return calls, simulate


def test_candidate_grid_is_stable_and_contains_identifiable_baseline() -> None:
    first = generate_baseline_long_only_candidates()
    second = generate_baseline_long_only_candidates()
    assert first == second
    assert len(first) == 60
    assert [item.parameters for item in first].count(
        BASELINE_LONG_ONLY_V1_PARAMETERS
    ) == 1
    assert sum(item.is_baseline for item in first) == 1


def test_split_is_chronological_disjoint_and_boundary_aligned() -> None:
    matrices, segments = split_feature_matrix(matrix())
    assert [item.decision_count for item in segments] == [6, 2, 2]
    all_boundaries = [
        row.candle_open_time + timedelta(minutes=5)
        for name in RefinementSegmentName
        for row in matrices[name].rows
    ]
    assert all_boundaries == sorted(all_boundaries)
    assert len(all_boundaries) == len(set(all_boundaries))
    assert all(value.minute in {0, 30} for value in all_boundaries)
    assert segments[0].last_boundary < segments[1].first_boundary
    assert segments[1].last_boundary < segments[2].first_boundary


def test_feature_matrix_is_built_once_for_all_candidates() -> None:
    calls, fake = fake_simulation()
    with (
        patch(
            "app.optimization.refinement.build_historical_feature_matrix",
            return_value=matrix(),
        ) as build,
        patch(
            "app.optimization.refinement.simulate_historical_feature_matrix",
            side_effect=fake,
        ),
    ):
        refine_historical_dataset(
            dataset=dataset(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
        )
    build.assert_called_once_with(dataset())
    assert calls


def test_training_top_k_validation_selection_and_test_are_separated() -> None:
    calls, fake = fake_simulation()
    with patch(
        "app.optimization.refinement.simulate_historical_feature_matrix",
        side_effect=fake,
    ):
        result = refine_historical_strategy(
            dataset=dataset(),
            feature_matrix=matrix(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            validation_top_k=3,
        )
    assert result.instrument is INSTRUMENT
    assert result.validation_candidate_count == 3
    assert result.training_ranking[0].candidate.parameters == (
        result.recommended_parameters
    )
    assert len(
        [call for call in calls if call[0] is RefinementSegmentName.VALIDATION]
    ) in {3, 4}
    assert result.recommended_parameters is not None
    assert result.recommended_parameters.entry_threshold == Decimal("0.70")
    assert (
        result.recommendation_status
        is RecommendationStatus.RECOMMENDED_AND_TEST_CONFIRMED
    )
    assert all(call[2:] == (Decimal("0.001"), Decimal("0.0005")) for call in calls)
    assert "validation net return" in result.recommendation_explanation
    assert "trades changed" in result.recommendation_explanation
    assert "max drawdown changed" in result.recommendation_explanation
    assert "historical evidence" in result.recommendation_explanation.lower()


def test_training_ranking_uses_return_drawdown_trades_then_stable_parameters() -> None:
    candidates = generate_baseline_long_only_candidates()[:4]
    results = (
        CandidateSegmentResult(
            candidates[0],
            RefinementSegmentName.TRAINING,
            Decimal("0.2"),
            Decimal("0.2"),
            2,
            Decimal(0),
            Decimal("100"),
        ),
        CandidateSegmentResult(
            candidates[1],
            RefinementSegmentName.TRAINING,
            Decimal("0.3"),
            Decimal("0.3"),
            1,
            Decimal(0),
            Decimal("100"),
        ),
        CandidateSegmentResult(
            candidates[2],
            RefinementSegmentName.TRAINING,
            Decimal("0.3"),
            Decimal("0.1"),
            3,
            Decimal(0),
            Decimal("100"),
        ),
        CandidateSegmentResult(
            candidates[3],
            RefinementSegmentName.TRAINING,
            Decimal("0.3"),
            Decimal("0.1"),
            2,
            Decimal(0),
            Decimal("100"),
        ),
    )
    ranked = tuple(sorted(results, key=_ranking_key))
    assert ranked == (results[3], results[2], results[1], results[0])


def test_refinement_result_is_exactly_deterministic() -> None:
    results = []
    for _ in range(2):
        _, fake = fake_simulation()
        with patch(
            "app.optimization.refinement.simulate_historical_feature_matrix",
            side_effect=fake,
        ):
            results.append(
                refine_historical_strategy(
                    dataset=dataset(),
                    feature_matrix=matrix(),
                    initial_cash=Decimal("10000"),
                    fee_rate=Decimal("0.001"),
                    slippage_rate=Decimal("0.0005"),
                    validation_top_k=3,
                )
            )
    assert results[0] == results[1]


def test_each_segment_restarts_with_initial_cash_and_closed_position() -> None:
    score = SimpleNamespace(overall_score=Decimal("0.8"))
    with patch(
        "app.optimization.simulation.calculate_baseline_score", return_value=score
    ):
        result = refine_historical_strategy(
            dataset=dataset(),
            feature_matrix=matrix(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            validation_top_k=1,
        )
    assert (
        result.baseline.training.trade_count,
        result.baseline.validation.trade_count,
        result.baseline.test.trade_count,
    ) == (1, 1, 1)


def test_test_result_does_not_change_selected_candidate() -> None:
    selected = []
    statuses = []
    for test_return in (Decimal("0.03"), Decimal("-0.03")):
        _, fake = fake_simulation(test_candidate_return=test_return)
        with patch(
            "app.optimization.refinement.simulate_historical_feature_matrix",
            side_effect=fake,
        ):
            result = refine_historical_strategy(
                dataset=dataset(),
                feature_matrix=matrix(),
                initial_cash=Decimal("10000"),
                fee_rate=Decimal("0.001"),
                slippage_rate=Decimal("0.0005"),
                validation_top_k=3,
            )
        selected.append(result.recommended_parameters)
        statuses.append(result.recommendation_status)
    assert selected[0] == selected[1]
    assert statuses == [
        RecommendationStatus.RECOMMENDED_AND_TEST_CONFIRMED,
        RecommendationStatus.RECOMMENDED_BUT_TEST_FAILED,
    ]


@pytest.mark.parametrize(
    ("validation_positive", "validation_trades"),
    [(False, 2), (True, 0)],
)
def test_negative_or_no_trade_validation_returns_no_recommendation(
    validation_positive: bool, validation_trades: int
) -> None:
    _, fake = fake_simulation(
        validation_positive=validation_positive,
        validation_trades=validation_trades,
    )
    with patch(
        "app.optimization.refinement.simulate_historical_feature_matrix",
        side_effect=fake,
    ):
        result = refine_historical_strategy(
            dataset=dataset(),
            feature_matrix=matrix(),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            validation_top_k=3,
        )
    assert result.recommended_parameters is None
    assert result.recommendation_status is RecommendationStatus.NO_VALIDATION_CANDIDATE


def test_too_few_boundaries_are_rejected() -> None:
    with pytest.raises(HistoricalRefinementError, match="five decision boundaries"):
        split_feature_matrix(HistoricalFeatureMatrix("fingerprint", matrix().rows[:4]))
