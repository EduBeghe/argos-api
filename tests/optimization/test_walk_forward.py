from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest

import app.optimization.walk_forward as walk_forward_module
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
    DEFAULT_REFINEMENT_BUDGET,
    RefinementBudget,
    RefinementCandidate,
    candidate_for,
)
from app.optimization.walk_forward import (
    WalkForwardBenchmarkEvaluation,
    WalkForwardCandidateEvaluation,
    WalkForwardCandidateSelection,
    WalkForwardClassification,
    WalkForwardConclusion,
    WalkForwardConfiguration,
    WalkForwardEvaluationStatus,
    WalkForwardMetrics,
    WalkForwardWindow,
    _aggregate,
    build_walk_forward_windows,
    build_window_matrices,
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
BUDGET = RefinementBudget(1, 1, 1, 1)


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
        refinement_budget=DEFAULT_REFINEMENT_BUDGET,
    )

    assert len(windows) == 3
    assert windows[0].ordinal == 1
    assert windows[0].training_from == START + timedelta(days=20)
    assert windows[0].training_to == START + timedelta(days=30)
    assert windows[0].validation_from == windows[0].training_to
    assert windows[0].test_from == windows[0].validation_to
    assert windows[0].test_to == START + timedelta(days=40)
    assert windows[1].training_from == START + timedelta(days=25)
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
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        configuration=configuration,
        refinement_budget=DEFAULT_REFINEMENT_BUDGET,
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
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        configuration=configuration,
        refinement_budget=DEFAULT_REFINEMENT_BUDGET,
    )
    second = evaluate_walk_forward(
        dataset=dataset,
        feature_matrix=matrix,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        configuration=configuration,
        refinement_budget=DEFAULT_REFINEMENT_BUDGET,
    )

    assert first == second
    assert first.conclusion is WalkForwardConclusion.COMPLETED
    assert len(first.windows) == 3
    assert all(
        item.window.test_from < item.window.test_to
        for item in first.candidate_evaluations
    )
    assert first.evaluations_computed > 0


def test_only_most_recent_windows_are_retained() -> None:
    dataset, matrix = _dataset_and_matrix(days=40)
    configuration = WalkForwardConfiguration(
        training_days=10,
        validation_days=5,
        test_days=5,
        step_days=5,
        maximum_windows=2,
        maximum_candidates=1,
        minimum_valid_windows=2,
    )

    windows = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=matrix,
        configuration=configuration,
        refinement_budget=BUDGET,
    )

    assert [window.ordinal for window in windows] == [1, 2]
    assert windows[0].training_from == START + timedelta(days=15)
    assert windows[-1].test_to == START + timedelta(days=40)


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
    selections = tuple(
        WalkForwardCandidateSelection(
            window,
            candidate,
            1,
            True,
            "selected_by_window_validation_rank",
            _metrics(value),
        )
        for window, value in zip(windows, returns, strict=True)
    )
    buy_hold_benchmarks = tuple(
        WalkForwardBenchmarkEvaluation(
            window,
            "buy_and_hold",
            WalkForwardEvaluationStatus.COMPLETED,
            _metrics(Decimal("0")),
            None,
        )
        for window in windows
    )
    production_benchmarks = tuple(
        WalkForwardBenchmarkEvaluation(
            window,
            "production_baseline",
            WalkForwardEvaluationStatus.COMPLETED,
            _metrics(Decimal("0")),
            None,
        )
        for window in windows
    )
    benchmarks = (*production_benchmarks, *buy_hold_benchmarks)
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
        selections,
        (*baseline_evaluations, *candidate_evaluations),
        benchmarks,
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
        refinement_budget=BUDGET,
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
        refinement_budget=BUDGET,
    )

    assert tuple(item.dataset_fingerprint for item in first) != tuple(
        item.dataset_fingerprint for item in second
    )


def test_window_matrices_are_half_open_and_test_evidence_cannot_change_selection() -> (
    None
):
    dataset, matrix = _dataset_and_matrix(days=25)
    configuration = WalkForwardConfiguration(
        training_days=10,
        validation_days=5,
        test_days=5,
        step_days=5,
        maximum_windows=1,
        maximum_candidates=1,
        minimum_valid_windows=1,
    )
    original = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=matrix,
        configuration=configuration,
        refinement_budget=BUDGET,
    )[0]
    matrices = build_window_matrices(matrix, original)

    assert all(
        original.training_from <= row.candle_open_time < original.training_to
        for row in matrices.training.rows
    )
    assert all(
        original.validation_from <= row.candle_open_time < original.validation_to
        for row in matrices.validation.rows
    )
    assert all(
        original.test_from <= row.candle_open_time < original.test_to
        for row in matrices.test.rows
    )
    assert not (
        {row.candle_open_time for row in matrices.training.rows}
        & {row.candle_open_time for row in matrices.validation.rows}
    )
    assert not (
        {row.candle_open_time for row in matrices.validation.rows}
        & {row.candle_open_time for row in matrices.test.rows}
    )

    changed_rows = tuple(
        _replace_close(row, Decimal("999999"))
        if original.test_from <= row.candle_open_time < original.test_to
        else row
        for row in matrix.rows
    )
    changed = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=HistoricalFeatureMatrix(
            matrix.dataset_fingerprint, changed_rows
        ),
        configuration=configuration,
        refinement_budget=BUDGET,
    )[0]

    assert changed.selection_fingerprint == original.selection_fingerprint
    assert changed.dataset_fingerprint != original.dataset_fingerprint


def test_later_window_evidence_cannot_change_earlier_selection_fingerprint() -> None:
    dataset, matrix = _dataset_and_matrix(days=30)
    configuration = WalkForwardConfiguration(
        training_days=10,
        validation_days=5,
        test_days=5,
        step_days=5,
        maximum_windows=2,
        maximum_candidates=1,
        minimum_valid_windows=1,
    )
    original = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=matrix,
        configuration=configuration,
        refinement_budget=BUDGET,
    )
    changed_rows = tuple(
        _replace_close(row, Decimal("888888"))
        if row.candle_open_time >= original[0].validation_to
        else row
        for row in matrix.rows
    )
    changed = build_walk_forward_windows(
        dataset=dataset,
        feature_matrix=HistoricalFeatureMatrix(
            matrix.dataset_fingerprint, changed_rows
        ),
        configuration=configuration,
        refinement_budget=BUDGET,
    )

    assert changed[0].selection_fingerprint == original[0].selection_fingerprint


def test_unselected_candidate_is_not_counted_as_failed_test_evidence() -> None:
    candidate = candidate_for(
        BASELINE_TECHNICAL_V1_PARAMETERS, BASELINE_LONG_ONLY_V1_PARAMETERS
    )
    window = _window(1)
    selection = WalkForwardCandidateSelection(
        window,
        candidate,
        2,
        False,
        "not_selected_by_window_validation_rank",
        _metrics(Decimal("0.01")),
    )
    stability = _aggregate(
        candidate,
        (window,),
        (selection,),
        (),
        (),
        WalkForwardConfiguration(minimum_valid_windows=1),
    )

    assert stability.classification is WalkForwardClassification.INSUFFICIENT_EVIDENCE
    assert stability.windows_evaluated == 0


def test_future_test_values_cannot_change_window_selection_or_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, matrix = _dataset_and_matrix(days=25)
    configuration = WalkForwardConfiguration(
        training_days=10,
        validation_days=5,
        test_days=5,
        step_days=5,
        maximum_windows=1,
        maximum_candidates=1,
        minimum_valid_windows=1,
    )
    score_a = BASELINE_TECHNICAL_V1_PARAMETERS
    score_b = BaselineTechnicalParameters(
        Decimal("0.4"),
        Decimal("0.2"),
        Decimal("0.2"),
        Decimal("0.2"),
        Decimal("0.1"),
    )
    monkeypatch.setattr(
        walk_forward_module,
        "generate_score_configurations",
        lambda: (score_a, score_b),
    )
    monkeypatch.setattr(
        walk_forward_module,
        "generate_strategy_configurations",
        lambda: (BASELINE_LONG_ONLY_V1_PARAMETERS,),
    )

    def metrics_for_candidate(*args: object) -> WalkForwardMetrics:
        candidate = args[-1]
        assert isinstance(candidate, RefinementCandidate)
        return _metrics(candidate.score_parameters.trend_weight)

    monkeypatch.setattr(walk_forward_module, "_simulate_metrics", metrics_for_candidate)
    budget = RefinementBudget(2, 2, 2, 2)
    first = evaluate_walk_forward(
        dataset=dataset,
        feature_matrix=matrix,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        configuration=configuration,
        refinement_budget=budget,
    )
    test_from = first.windows[0].test_from
    altered = HistoricalFeatureMatrix(
        matrix.dataset_fingerprint,
        tuple(
            _replace_close(row, Decimal("999999"))
            if row.candle_open_time >= test_from
            else row
            for row in matrix.rows
        ),
    )
    second = evaluate_walk_forward(
        dataset=dataset,
        feature_matrix=altered,
        initial_cash=Decimal("10000"),
        fee_rate=Decimal("0.001"),
        slippage_rate=Decimal("0.0005"),
        configuration=configuration,
        refinement_budget=budget,
    )

    first_identity = tuple(
        (
            item.candidate.stable_candidate_id,
            item.validation_rank,
            item.selected_for_test,
        )
        for item in first.selections
    )
    second_identity = tuple(
        (
            item.candidate.stable_candidate_id,
            item.validation_rank,
            item.selected_for_test,
        )
        for item in second.selections
    )
    assert first_identity == second_identity


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


def _replace_close(
    row: HistoricalFeatureRow, close_price: Decimal
) -> HistoricalFeatureRow:
    return HistoricalFeatureRow(
        candle_open_time=row.candle_open_time,
        close_price=close_price,
        ema_20=row.ema_20,
        ema_50=row.ema_50,
        rsi_14=row.rsi_14,
        atr_14=row.atr_14,
        return_1=row.return_1,
        return_12=row.return_12,
        relative_volume_20=row.relative_volume_20,
    )


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
