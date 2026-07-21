from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

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
from app.optimization.refinement import RefinementCandidate, candidate_for
from app.optimization.walk_forward import (
    WalkForwardBenchmarkEvaluation,
    WalkForwardCandidateEvaluation,
    WalkForwardClassification,
    WalkForwardConclusion,
    WalkForwardConfiguration,
    WalkForwardEvaluationStatus,
    WalkForwardMetrics,
    WalkForwardWindow,
    _aggregate,
    build_walk_forward_windows,
    evaluate_walk_forward,
)
from app.scores.calculator import (
    BASELINE_TECHNICAL_V1_PARAMETERS,
    BaselineTechnicalParameters,
)
from app.strategies.baseline_long_only import (
    BASELINE_LONG_ONLY_V1_PARAMETERS,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
INSTRUMENT = Instrument(
    Asset.BTC_USDT,
    Provider.BINANCE,
    MarketType.SPOT,
    Interval.FIVE_MINUTES,
)


def test_windows_are_half_open_chronological_and_bounded() -> None:
    dataset, matrix = _dataset_and_matrix(days=50)
    configuration = WalkForwardConfiguration(
        training_days=10,
        validation_days=5,
        test_days=5,
        step_days=5,
        maximum_windows=3,
        maximum_candidates=2,
        minimum_valid_windows=2,
    )

    windows = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=matrix,
        configuration=configuration,
    )

    assert len(windows) == 3
    assert windows[0].training_from == START
    assert windows[0].training_to == START + timedelta(days=10)
    assert windows[0].validation_from == windows[0].training_to
    assert windows[0].test_from == windows[0].validation_to
    assert windows[0].test_to == START + timedelta(days=20)
    assert windows[1].training_from == START + timedelta(days=5)
    assert windows[0].test_to <= windows[1].test_to
    assert all(window.decision_count == 5 for window in windows)
    assert len({window.dataset_fingerprint for window in windows}) == 3


def test_short_dataset_returns_explicit_insufficient_history() -> None:
    dataset, matrix = _dataset_and_matrix(days=19)
    configuration = WalkForwardConfiguration(
        training_days=10,
        validation_days=5,
        test_days=5,
        step_days=5,
        maximum_windows=3,
        maximum_candidates=1,
        minimum_valid_windows=2,
    )

    result = evaluate_walk_forward(
        dataset=dataset,
        feature_matrix=matrix,
        candidates=(),
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        configuration=configuration,
    )

    assert result.conclusion is WalkForwardConclusion.INSUFFICIENT_HISTORY
    assert result.windows == ()
    assert result.recommended_candidate_id is None
    assert "No walk-forward recommendation" in result.recommendation_explanation


def test_evaluation_is_deterministic_and_uses_only_test_rows() -> None:
    dataset, matrix = _dataset_and_matrix(days=40)
    configuration = WalkForwardConfiguration(
        training_days=10,
        validation_days=5,
        test_days=5,
        step_days=5,
        maximum_windows=3,
        maximum_candidates=1,
        minimum_valid_windows=2,
    )
    first = evaluate_walk_forward(
        dataset=dataset,
        feature_matrix=matrix,
        candidates=(),
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        configuration=configuration,
    )
    second = evaluate_walk_forward(
        dataset=dataset,
        feature_matrix=matrix,
        candidates=(),
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        configuration=configuration,
    )

    assert first == second
    assert first.conclusion is WalkForwardConclusion.COMPLETED
    assert len(first.windows) == 3
    assert all(
        item.window.test_from < item.window.test_to
        for item in first.candidate_evaluations
    )
    assert first.evaluations_computed == 6


def test_candidate_limit_and_uniqueness_are_enforced() -> None:
    dataset, matrix = _dataset_and_matrix(days=40)
    candidate = candidate_for(
        BaselineTechnicalParameters(
            Decimal("0.4"),
            Decimal("0.2"),
            Decimal("0.2"),
            Decimal("0.2"),
            Decimal("0.1"),
        ),
        BASELINE_LONG_ONLY_V1_PARAMETERS,
    )
    configuration = WalkForwardConfiguration(
        training_days=10,
        validation_days=5,
        test_days=5,
        step_days=5,
        maximum_windows=2,
        maximum_candidates=1,
        minimum_valid_windows=2,
    )

    with pytest.raises(ValueError, match="candidate count exceeds"):
        evaluate_walk_forward(
            dataset=dataset,
            feature_matrix=matrix,
            candidates=(candidate, candidate),
            initial_cash=Decimal("10000"),
            fee_rate=Decimal("0.001"),
            slippage_rate=Decimal("0.0005"),
            configuration=configuration,
        )


@pytest.mark.parametrize(
    ("returns", "expected"),
    [
        (
            (Decimal("0.03"), Decimal("0.02"), Decimal("0.01")),
            WalkForwardClassification.STABLE_CANDIDATE,
        ),
        (
            (Decimal("1.00"), Decimal("-0.01"), Decimal("-0.01")),
            WalkForwardClassification.UNSTABLE_CANDIDATE,
        ),
        (
            (Decimal("-0.03"), Decimal("-0.02"), Decimal("-0.01")),
            WalkForwardClassification.CONSISTENTLY_UNPROFITABLE,
        ),
    ],
)
def test_classification_requires_multi_window_stability(
    returns: tuple[Decimal, ...], expected: WalkForwardClassification
) -> None:
    baseline = candidate_for(
        BASELINE_TECHNICAL_V1_PARAMETERS, BASELINE_LONG_ONLY_V1_PARAMETERS
    )
    candidate = candidate_for(
        BaselineTechnicalParameters(
            Decimal("0.4"),
            Decimal("0.2"),
            Decimal("0.2"),
            Decimal("0.2"),
            Decimal("0.1"),
        ),
        BASELINE_LONG_ONLY_V1_PARAMETERS,
    )
    windows = tuple(_window(index) for index in range(1, 4))
    baseline_evaluations = tuple(
        _evaluation(window, baseline, Decimal("0")) for window in windows
    )
    candidate_evaluations = tuple(
        _evaluation(window, candidate, value)
        for window, value in zip(windows, returns, strict=True)
    )
    benchmarks = tuple(
        WalkForwardBenchmarkEvaluation(
            window,
            "buy_and_hold",
            WalkForwardEvaluationStatus.COMPLETED,
            _metrics(Decimal("0")),
            None,
        )
        for window in windows
    )
    configuration = WalkForwardConfiguration(
        training_days=1,
        validation_days=1,
        test_days=1,
        step_days=1,
        maximum_windows=3,
        maximum_candidates=2,
        minimum_valid_windows=3,
        minimum_profitable_window_ratio=Decimal("0.60"),
        maximum_drawdown=Decimal("0.20"),
        maximum_return_standard_deviation=Decimal("0.20"),
    )

    value = _aggregate(
        candidate,
        windows,
        (*baseline_evaluations, *candidate_evaluations),
        benchmarks,
        baseline.stable_candidate_id,
        configuration,
    )

    assert value.classification is expected
    assert value.windows_evaluated == 3
    expected_ratio = (Decimal(sum(item > 0 for item in returns)) / Decimal(3)).quantize(
        Decimal("0.000000000000000001")
    )
    assert value.positive_excess_return_vs_production_ratio == expected_ratio


def test_configuration_change_changes_window_fingerprints() -> None:
    dataset, matrix = _dataset_and_matrix(days=50)
    first = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=matrix,
        configuration=WalkForwardConfiguration(
            training_days=10,
            validation_days=5,
            test_days=5,
            step_days=5,
            maximum_windows=2,
            maximum_candidates=1,
            minimum_valid_windows=2,
        ),
    )
    second = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=matrix,
        configuration=WalkForwardConfiguration(
            training_days=11,
            validation_days=5,
            test_days=5,
            step_days=5,
            maximum_windows=2,
            maximum_candidates=1,
            minimum_valid_windows=2,
        ),
    )

    assert tuple(item.dataset_fingerprint for item in first) != tuple(
        item.dataset_fingerprint for item in second
    )


def _dataset_and_matrix(
    *, days: int
) -> tuple[HistoricalDataset, HistoricalFeatureMatrix]:
    end = START + timedelta(days=days)
    dataset = HistoricalDataset(
        identity=HistoricalDatasetIdentity(
            INSTRUMENT, HistoricalDatasetRange(START, end)
        ),
        expected_candle_count=0,
        actual_candle_count=0,
        first_open_time=None,
        last_open_time=None,
        gap_count=0,
        first_gap=None,
        last_gap=None,
        candles=(),
        fingerprint=f"dataset-{days}",
    )
    rows = tuple(
        HistoricalFeatureRow(
            candle_open_time=START + timedelta(days=index, minutes=25),
            close_price=Decimal("100") + Decimal(index),
            ema_20=Decimal("100"),
            ema_50=Decimal("99"),
            rsi_14=Decimal("55"),
            atr_14=Decimal("1"),
            return_1=Decimal("0.01"),
            return_12=Decimal("0.02"),
            relative_volume_20=Decimal("1"),
        )
        for index in range(days)
    )
    return dataset, HistoricalFeatureMatrix(dataset.fingerprint, rows)


def _window(ordinal: int) -> WalkForwardWindow:
    start = START + timedelta(days=ordinal - 1)
    return WalkForwardWindow(
        ordinal,
        start,
        start + timedelta(hours=6),
        start + timedelta(hours=6),
        start + timedelta(hours=12),
        start + timedelta(hours=12),
        start + timedelta(days=1),
        1,
        str(uuid4()),
    )


def _metrics(total_return: Decimal) -> WalkForwardMetrics:
    return WalkForwardMetrics(
        total_return=total_return,
        absolute_pnl=total_return * Decimal("10000"),
        max_drawdown=Decimal("0.05"),
        final_equity=Decimal("10000") * (Decimal(1) + total_return),
        fees_paid=Decimal("10"),
        trade_count=2,
        return_over_drawdown=total_return / Decimal("0.05"),
    )


def _evaluation(
    window: WalkForwardWindow,
    candidate: RefinementCandidate,
    total_return: Decimal,
) -> WalkForwardCandidateEvaluation:
    return WalkForwardCandidateEvaluation(
        window,
        candidate,
        WalkForwardEvaluationStatus.COMPLETED,
        _metrics(total_return),
        None,
    )
